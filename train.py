import json
import os
from json import dumps

import dgl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import utils
from args import get_args
from data.dataset import ReadmissionDataset
from model.model import GraphRNN, GConvLayers
from model.simple_lstm import SimpleLSTM
from model.simple_rnn import SimpleRNN


def auc_ci(y_true, y_pred, num_bootstraps=1000, ci=95):
    bootstrap_means = torch.empty(num_bootstraps)

    for i in range(num_bootstraps):
        indices = torch.randint(0, len(y_pred), (len(y_pred),))
        bootstrap_means[i] = roc_auc_score(y_true[indices], y_pred[indices])

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = bootstrap_means.quantile(lower_percentile / 100)
    upper_bound = bootstrap_means.quantile(upper_percentile / 100)

    return lower_bound.item(), upper_bound.item()


def evaluate(
        args,
        model,
        graph,
        features,
        labels,
        nid,
        loss_fn,
        best_thresh=0.5,
        save_file=None,
        thresh_search=False,
        device="cpu",
        evaluate_ci=False,
):
    model.eval()
    with torch.no_grad():
        dataset = TensorDataset(features[nid], labels[nid])
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
        preds, losses = [], []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs).squeeze()
            loss = loss_fn(logits, targets)

            losses.append(loss.item())
            preds.append(logits.cpu())

        loss = torch.tensor(losses).mean()

        logits = torch.cat(preds, dim=0)
        probs = torch.sigmoid(logits)

        preds = (probs >= best_thresh).int().numpy()

        eval_results = utils.eval_dict(
            y=labels[nid].numpy(),
            y_pred=preds,
            y_prob=probs.numpy(),
            average="binary",
            thresh_search=thresh_search,
            best_thresh=best_thresh,
        )
        eval_results["loss"] = loss.item()

        if evaluate_ci:
            lower_bound, upper_bound = auc_ci(labels[nid], probs, num_bootstraps=1000)
            eval_results["ci_lower"] = lower_bound
            eval_results["ci_upper"] = upper_bound
    return eval_results


def main(args):
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # set random seed
    utils.seed_torch(seed=args.rand_seed)

    # get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False
    )

    # save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    logger = utils.get_logger(args.save_dir, "train")
    logger.info("Args: {}".format(dumps(vars(args), indent=4, sort_keys=True)))

    # load graph
    logger.info("Constructing graph...")
    dataset = ReadmissionDataset(
        demo_file=args.demo_file,
        edge_ehr_file=args.edge_ehr_file,
        ehr_feature_file=args.ehr_feature_file,
        edge_modality=args.edge_modality,
        top_perc=args.edge_top_perc,
        gauss_kernel=args.use_gauss_kernel,
        max_seq_len_ehr=args.max_seq_len_ehr,
        standardize=True,
        ehr_types=args.ehr_types,
        is_graph=False,
    )
    g = dataset[0]
    cat_idxs = dataset.cat_idxs
    cat_dims = dataset.cat_dims

    features = g.ndata["feat"]
    labels = g.ndata["label"].float()
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # ensure self-edges
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    logger.info(
        """----Graph Stats------
            # Nodes %d
            # Undirected edges %d
            # Average degree %d """
        % (
            n_nodes,
            int(n_edges / 2),
            g.in_degrees().float().mean().item(),
        )
    )

    train_nid = torch.nonzero(train_mask).squeeze()
    val_nid = torch.nonzero(val_mask).squeeze()
    test_nid = torch.nonzero(test_mask).squeeze()

    logger.info(
        "#Train samples: {:,}; positive percentage: {:.2%}".format(
            train_mask.sum(), labels[train_mask].mean()
        )
    )
    logger.info(
        "#Val samples: {:,}; positive percentage: {:.2%}".format(
            val_mask.sum(), labels[val_mask].mean()
        )
    )
    logger.info(
        "#Test samples: {:,}; positive percentage: {:.2%}".format(
            test_mask.sum(), labels[test_mask].mean(),
        )
    )

    if args.model_name == "stgnn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GraphRNN(
            in_dim=in_dim,
            n_classes=args.num_classes,
            device=device,
            is_classifier=True,
            ehr_encoder_name="embedder" if args.feature_type != "imaging" else None,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )

    elif args.model_name == "rnn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        model = SimpleRNN(
            input_size=in_dim,
            hidden_size=args.hidden_dim,
            output_size=1,
            num_layers=args.num_rnn_layers,
            dropout=args.dropout,
        )
    elif args.model_name == "lstm":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        model = SimpleLSTM(
            input_size=in_dim,
            hidden_size=args.hidden_dim,
            output_size=1,
            num_layers=args.num_rnn_layers,
            dropout=args.dropout,
        )
    else:
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GConvLayers(
            in_dim=in_dim,
            num_classes=args.num_classes,
            is_classifier=True,
            device=device,
            **config
        )

    model.to(device)
    print("Compiling model...")
    model = torch.compile(model)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_wd
    )

    # load model checkpoint
    if args.load_model_path is not None:
        model, optimizer = utils.load_model_checkpoint(
            args.load_model_path, model, optimizer
        )

    # count params
    params = utils.count_parameters(model)
    logger.info("Trainable parameters: {}".format(params))

    # loss func
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.pos_weight)).to(
        device
    )

    # checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=args.save_dir,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=logger,
    )

    # scheduler
    logger.info("Using cosine annealing scheduler...")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    if args.do_train:
        # Train
        logger.info("Training...")
        model.train()
        epoch = 0
        prev_val_loss = 1e10
        patience_count = 0
        early_stop = False

        while (epoch != args.num_epochs) and (not early_stop):

            epoch += 1
            logger.info("Starting epoch {}...".format(epoch))
            train_loss = []

            dataset = TensorDataset(features[train_nid], labels[train_nid])
            dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
            for inputs, targets in tqdm(dataloader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs).squeeze()
                loss = loss_fn(logits, targets)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            # evaluate on val set
            if epoch % args.eval_every == 0:
                logger.info("Evaluating at epoch {}...".format(epoch))
                eval_results = evaluate(
                    args=args,
                    model=model,
                    graph=g,
                    features=features,
                    labels=labels,
                    nid=val_nid,
                    loss_fn=loss_fn,
                    device=device,
                )
                model.train()
                saver.save(epoch, model, optimizer, eval_results[args.metric_name])
                # accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Log to console

                logger.info("TRAIN - Epoch: {} | Loss: {:.4f}".format(epoch, sum(train_loss) / len(
                    train_loss)))
                results_str = ", ".join(
                    "{}: {:.3f}".format(k, eval_results[k]) for k in ["auroc", "loss"]
                )
                logger.info("VAL - {}".format(results_str))

            # step lr scheduler
            scheduler.step()

        logger.info("Training DONE.")
        best_path = os.path.join(args.save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)
        model.to(device)

    # evaluate
    val_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=val_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "val_predictions.pkl"),
        thresh_search=args.thresh_search,
        device=device,
    )
    val_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in val_results.items()
    )
    logger.info("VAL - {}".format(val_results_str))

    # eval on test set
    test_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=test_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "test_predictions.pkl"),
        best_thresh=val_results["best_thresh"],
        device=device,
        evaluate_ci=True
    )
    test_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in test_results.items()
    )
    logger.info("TEST - {}".format(test_results_str))

    logger.info("Results saved to {}".format(args.save_dir))

    return val_results[args.metric_name]


if __name__ == "__main__":
    main(get_args())

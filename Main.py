# basic libs:
import torch
import timeit

# training options control panel:
from Arguments import parser

from libs.Train import train_base
from libs import Helpers


def main():
    global args
    args = parser.parse_args()

    # fix a random seed:
    Helpers.reproducibility(args)

    # model intialisation:
    model, model_name = Helpers.network_intialisation(args)

    # resume training:
    if args.resume is True:
        model = torch.load(args.checkpoint_path)

    # put model in the gpu:
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.l2)

    # data loaders:
    data_loaders = Helpers.get_data_simple_wrapper(args)

    # make saving directories:
    writer, saved_model_path = Helpers.make_saving_directories(model_name, args)

    # set up timer:
    start_time = timeit.default_timer()

    # train data loader:
    data_iterators = Helpers.get_iterators(data_loaders, args)
    # train labelled:
    train_labelled_data_loader = data_iterators.get('train labelled')
    iterator_train_labelled = iter(train_labelled_data_loader)

    # train unlabelled:
    if args.unlabelled > 0:
        train_unlabelled_data_loader = data_iterators.get('train unlabelled')
        iterator_train_unlabelled = iter(train_unlabelled_data_loader)

    # running loop:
    for step in range(args.iterations):
        # put model to training mode:
        model.train()

        if args.unlabelled > 0:
            # unlabelled data:
            unlabelled_dict = Helpers.get_data_dict(train_unlabelled_data_loader, iterator_train_unlabelled)
            labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        else:
            # labelled data
            labelled_dict = Helpers.get_data_dict(train_labelled_data_loader, iterator_train_labelled)

        losses = train_base(labelled_img=labelled_dict,
                            labelled_label=labelled_dict,
                            unlabelled_img=None,
                            model=model,
                            t=args.temp,
                            prior_mu=args.mu,
                            prior_logsigma=args.std,
                            augmentation_cutout=True)


















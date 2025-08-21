import os
import torch
from tqdm import tqdm
from network import Network
from utils.metric import valid
from torch.utils.data import Dataset
import argparse
from loss import Loss
from dataloader import load_data
from utils.train_utils import pretrain, contrastive_train
from utils.utils import setup_seed

# MNIST-USPS, BDGP, CCV, Fashion, Caltech-2V, Caltech-3V, Caltech-4V, Caltech-5V
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='CCV')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--selflabel_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--network", default='AE')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# lambda1 = 1.0 lambda2 = 10.0 {'acc': 0.9976, 'nmi': 0.9926, 'pur': 0.9976}
if args.dataset == "MNIST_USPS":
    args.pre_epochs = 0
    args.con_epochs = 50
    args.seed = 4
    args.lambda1 = 1.0
    args.lambda2 = 10.0
    args.network = 'CAE'
# lambda1 = 1.0 lambda2 = 1.0 {'acc': 0.9936, 'nmi': 0.9799360503452945, 'pur': 0.9936}
if args.dataset == "BDGP":
    args.con_epochs = 10
    args.seed = 0
    args.lambda1 = 1.0
    args.lambda2 = 1.0
# lambda1 = 1.0 lambda2 = 1.0 {'acc': 0.32969142182194006, 'nmi': 0.32339217661858716, 'pur': 0.3794478074708401}
if args.dataset == "CCV":
    args.con_epochs = 50
    args.seed = 10
    args.lambda1 = 1.0
    args.lambda2 = 1.0
    args.network = 'AE'
# lambda1 = 0.1 lambda2 = 0.1 {'acc': 0.9939, 'nmi': 0.9837702422487985, 'pur': 0.9939}
if args.dataset == "Fashion":
    args.pre_epochs = 0
    args.con_epochs = 100
    args.seed = 3
    args.lambda1 = 1.0
    args.lambda2 = 0.1
    args.network = 'CAE'
# lambda1 = 0.1 lambda2 = 0.1 {'acc': 0.67, 'nmi': 0.5441733894079993, 'pur': 0.67}
if args.dataset == "Caltech_2V":
    args.con_epochs = 50
    args.seed = 0
    args.lambda1 = 0.1
    args.lambda2 = 0.1
# lambda1 = 0.1 lambda2 = 0.1 {'acc': 0.7464285714285714, 'nmi': 0.6538771169445668, 'pur': 0.7478571428571429}
if args.dataset == "Caltech_3V":
    args.con_epochs = 50
    args.seed = 0
    args.lambda1 = 0.1
    args.lambda2 = 0.1
# lambda1 = 0.1 lambda2 = 0.1 {'acc': 0.8392857142857143, 'nmi': 0.7579481624026398, 'pur': 0.8392857142857143}
if args.dataset == "Caltech_4V":
    args.con_epochs = 50
    args.seed = 7
    args.lambda1 = 0.1
    args.lambda2 = 0.1
# lambda1 = 0.1 lambda2 = 0.1 {'acc': 0.8978571428571429, 'nmi': 0.8308507744607068, 'pur': 0.8978571428571429}
if args.dataset == "Caltech_5V":
    args.con_epochs = 50
    args.seed = 6
    args.lambda1 = 0.1
    args.lambda2 = 0.1

if __name__ == '__main__':
    setup_seed(args.seed)

    # Dataset
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Model
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device, args.network)
    model = model.to(device)
    print(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    temperature = [0.7, 0.9]
    for temp in temperature:
        args.temperature_f = temp
        # Pre_train phase
        if args.pre_epochs > 0:
            print('Pre_training ...')
            for i in tqdm(range(args.pre_epochs)):
                pretrain(i, data_loader, view, optimizer, model, device)

        # Contrastive phase
        print('Contrastive training ...')
        for i in tqdm(range(args.con_epochs)):
            contrastive_train(i, data_loader, view, optimizer, model, criterion, device, args)

        results = valid(model, device, dataset, view, data_size, class_num, eval_h=False)

    # # t-SNE
    # embeddings, Qs, labels = get_embedding(dataset, model, device, view, data_size)
    # tsne_plot(embeddings, labels, './results/tsne/')
    # metrics_plot(embeddings, './results/metrics')
    # metrics_plot(Qs, './results/metrics')

    # with open('./results/log.txt', 'a+') as fw:
    #     fw.write(args.dataset + ' ' + str(args.lambda1) + ' ' + str(args.lambda2) + ' ' + str(results) + '\n')
        import csv
        with open('./results/CCV_tau.csv', 'a+') as csvfile:
            writer = csv.writer(csvfile)
            if not (os.path.exists('./results/CCV_tau.csv') and os.path.getsize('./results/CCV_tau.csv') > 0):
                writer.writerow(['dataset', 'seed', 'tau', 'lambda1', 'lambda2', 'ACC', 'NMI', 'PUR'])
            writer.writerow([args.dataset, args.seed, args.temperature_f, args.lambda1, args.lambda2] + [results[key] for key in results])




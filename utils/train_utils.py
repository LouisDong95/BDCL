import torch


def pretrain(epoch, data_loader, view, optimizer, model, device):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch, data_loader, view, optimizer, model, criterion, device, args):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, qns, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_instance(hs[v], hs[w]) * args.lambda1)
                loss_list.append(criterion.forward_consistency(qs[v], qns[v], qs[w], qns[w]) * args.lambda1)
            loss_list.append(mes(xs[v], xrs[v]))
            loss_list.append(criterion.forward_feature(zs[v]) * args.lambda2)
            loss_list.append(criterion.forward_cluster(qs[v]) * args.lambda2)
            loss_list.append(criterion.entropy(qs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
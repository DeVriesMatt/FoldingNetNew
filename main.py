from dataset import *
from model import *
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from naming_and_reports import *
from datetime import date, time


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--dataset_path', default='./', type=str)
    parser.add_argument('--dataframe_path', default='./', type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--num_features', default=50, type=int)
    parser.add_argument('--shape', default='sphere', type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_dir = args.output_dir
    num_features = args.num_features
    shape = args.shape

    model_name = 'FoldingNetNew_{}feats_{}shape'.format(num_features, shape)
    f, name_net, saved_to, name_txt, name = reports(model_name, output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    to_eval = "ReconstructionNet" + "(" + "'{0}'".format("dgcnn_cls") + ", num_clusters=5, " \
                                                                        "num_features=num_features, " \
                                                                        "shape=shape)"
    model = eval(to_eval)
    model = model.to(device)

    dataset = PointCloudDataset(df,
                                root_dir,
                                transform=None,
                                img_size=400,
                                target_transform=True)

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_inf = DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Logging
    writer_logging = output_dir + 'runs/' + name + '_{}feats_{}shape'.format(num_features, shape)
    writer = SummaryWriter(log_dir=writer_logging)
    date_time = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_")[:-7]
    num_epochs = 500

    print_both(f, 'Date: ' + date_time)
    print_both(f, 'Training model: ' + name)
    print_both(f, 'Number of epochs: {}'.format(num_epochs))
    print_both(f, 'Number of features: {}'.format(num_features))
    print_both(f, 'Shape used: {}'.format(shape))
    print_both(f, 'Output directory of nets: ' + name_net + '.pt')
    print_both(f, 'Output directory of reports ' + name_txt)
    print_both(f, 'Output directory of tensorboard logging ' + writer_logging)
    f.close()

    total_loss = 0.
    rec_loss = 0.
    clus_loss = 0.
    model.train()
    threshold = 0.
    losses = []
    test_acc = []
    best_acc = 0.
    best_loss = 1000000000
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.
        f = open(name_txt, 'a')
        print_both(f, 'Training epoch {}'.format(epoch))
        f.close()
        model.train()
        batches = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output, feature, embedding, clustering_out, fold1 = model(inputs)

                loss = model.get_loss(inputs, output)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += float(loss)/batch_size
            batch_num += 1
            writer.add_scalar('/Loss' + 'Batch', loss.item()/batch_size, (i + 1) * (epoch + 1))
            lr = np.asarray(optimizer.param_groups[0]['lr'])

            if i % 10 == 0:
                f = open(name_txt, 'a')
                print_both(f, '[%d/%d][%d/%d]\tLossTot: %.4f\tLossRec: %.4f' % (epoch,
                                                                                num_epochs,
                                                                                i,
                                                                                len(dataloader),
                                                                                loss.item()/batch_size,
                                                                                loss.item()/batch_size,))

                f.close()
        # ===================log========================
        total_loss = running_loss/len(dataloader)
        if total_loss < best_loss:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': running_loss}
            best_loss = total_loss
            create_dir_if_not_exist(output_dir)
            f = open(name_txt, 'a')
            print_both(f, 'Saving model to:' + name_net + '.pt' + ' with loss = {}'
                  .format(total_loss) + ' at epoch {}'.format(epoch))
            torch.save(checkpoint, name_net + '.pt')
            print_both(f, 'epoch [{}/{}], loss:{}'.format(epoch + 1, num_epochs, total_loss))
            f.close()

        f = open(name_txt, 'a')
        print_both(f, 'epoch [{}/{}], loss:{:.4f}, Rec loss:{:.4f}'.format(epoch + 1,
                                                                           num_epochs,
                                                                           total_loss,
                                                                           total_loss))
        f.close()


from dataset import *
from model import *
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--dataset_path', default='./', type=str)
    parser.add_argument('--dataframe_path', default='./', type=str)
    parser.add_argument('--output_dir', default='./', type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_dir = args.output_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    to_eval = "ReconstructionNet" + "(" + "'{0}'".format("dgcnn_cls") + ", num_clusters=5)"
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter()
    date_time = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_")[:-7]
    num_epochs = 500
    total_loss = 0.
    rec_loss = 0.
    clus_loss = 0.
    model.train()
    threshold = 0.
    losses = []
    test_acc = []
    best_acc = 0.
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.
        previous_loss = 1000000000
        print('Training epoch {}'.format(epoch))
        model.train()
        batches = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output, feature, embedding, clustering_out, fold1 = model(inputs)

                loss_rec = model.get_loss(inputs, output)

                loss = loss_rec
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += float(loss)
            batch_num += 1
            writer.add_scalar('/Loss' + 'Batch', loss.item(), (i + 1) * (epoch + 1))
            lr = np.asarray(optimizer.param_groups[0]['lr'])

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLossTot: %.4f\tLossRec: %.4f'
                      % (epoch, num_epochs, i, len(dataloader), loss.item(),
                         loss_rec.item(),))

        # ===================log========================
        if running_loss < previous_loss:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': running_loss}
            torch.save(checkpoint, output_dir + 'AE_spheroid_nuclei.pt')
        total_loss += float(loss)

        #     scheduler.step(total_loss)
        print('epoch [{}/{}], loss:{}'
              .format(epoch + 1, num_epochs, total_loss))

        total_loss += float(loss)
        rec_loss += float(loss_rec)
        print('epoch [{}/{}], loss:{:.4f}, Rec loss:{:.4f}'
              .format(epoch + 1, num_epochs, total_loss, rec_loss))


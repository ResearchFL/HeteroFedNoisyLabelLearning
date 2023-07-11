import copy
import numpy as np
import torch
from model.build_model import build_model
import sys
sys.path.append("..")
from model.build_model import build_model
from loss import SCELoss
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import torch.nn as nn
import numpy as np
from util.load_data import load_data_with_noisy_label
from numpy import *
from util.local_training import globaltest, get_local_update_objects
import time


def RFL(args):
    ##############################
    #  Load Dataset
    ##############################
    dataset_train, dataset_test, dict_users, y_train, gamma_s = load_data_with_noisy_label(args)

    start = time.time()
    ##############################
    # Build model
    ##############################
    net_glob = build_model(args)
    print(net_glob)
    col_loss_list = []
    local_loss_list = []
    acc_list = []
    current_mean_loss_list = [] # for CCR reweight
    for epoch_index in range(args.rounds2):

        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname = Private_Nets_Name_List[participant_index]
            private_dataset_dir = Private_Dataset_Dir
            # print(netname + '_' + Private_Dataset_Name + '_' + private_dataset_dir)
            _, test_dl, _, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=private_dataset_dir,
                                              train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None, noise_type=Noise_type, noise_rate=Noise_rate)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants

        '''
        Calculate Client Confidence with label quality and model performance
        '''
        amount_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        weight_with_quality = []
        quality_list = []
        amount_with_quality_exp = []
        last_mean_loss_list = current_mean_loss_list
        current_mean_loss_list = []
        delta_mean_loss_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                                  train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                                  dataidxs=private_dataidx, noise_type=Noise_type, noise_rate=Noise_rate)
            if Client_Confidence_Reweight_Loss == 'CE':
                criterion = nn.CrossEntropyLoss()
            if Client_Confidence_Reweight_Loss == 'SCE':
                criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
            criterion.to(device)
            participant_loss_list = []
            for batch_idx, (images, labels) in enumerate(train_dl_local):
                images = images.to(device)
                labels = labels.to(device)
                private_linear_output, _ = network(images)
                private_loss = criterion(private_linear_output, labels)
                participant_loss_list.append(private_loss.item())
            mean_participant_loss = mean(participant_loss_list)
            current_mean_loss_list.append(mean_participant_loss)
        #EXP标准化处理
        if epoch_index > 0 :
            for participant_index in range(N_Participants):
                delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
                quality_list.append(delta_loss / current_mean_loss_list[participant_index])
            quality_sum = sum(quality_list)
            for participant_index in range(N_Participants):
                amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
                amount_with_quality_exp.append(exp(amount_with_quality[participant_index]))
            amount_with_quality_sum = sum(amount_with_quality_exp)
            for participant_index in range(N_Participants):
                weight_with_quality.append(amount_with_quality_exp[participant_index] / amount_with_quality_sum)
        else:
            weight_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        weight_with_quality = torch.tensor(weight_with_quality)

        '''
        HHF
        '''
        for batch_idx, (images, _) in enumerate(public_train_dl):
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            '''
            Calculate Linear Output
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                images = images.to(device)
                linear_output,_ = network(x=images)
                linear_output_softmax = F.softmax(linear_output,dim =1)
                linear_output_target_list.append(linear_output_softmax.clone().detach())
                linear_output_logsoft = F.log_softmax(linear_output,dim=1)
                linear_output_list.append(linear_output_logsoft)
            '''
            Update Participants' Models via KL Loss and Data Quality
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()
                loss = torch.tensor(0)
                for i in range(N_Participants):
                    if i != participant_index:
                        weight_index = weight_with_quality[i]
                        loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                        temp = weight_index * loss_batch_sample
                        loss = loss + temp
                kl_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()
            col_loss_list.append(kl_loss_batch_list)

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,
            train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=private_dataidx, noise_type=Noise_type, noise_rate=Noise_rate)
            private_epoch = max(int(len(public_train_ds)/len(train_ds_local)),1)

            network,private_loss_batch_list = update_model_via_private_data(network=network,private_epoch=private_epoch,
            private_dataloader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
            optimizer_method=Pariticpant_Params['optimizer_name'],learing_rate=Pariticpant_Params['learning_rate'],
            logger=logger)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        """
        Evaluate ModelS in the final round
        """
        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):  # 改成2 拿来测试 N_Participants
                _, test_dl, _, _ = get_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None, noise_type=Noise_type, noise_rate=Noise_rate)
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / N_Participants

        if epoch_index % 5 == 0 or epoch_index==CommunicationEpoch-1:
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction'])
            mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction'])
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
            mkdirs('./test/Model_Storage/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
            mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+str(Noise_rate))
            mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+ str(Noise_rate))

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
                    +'/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type) +str(Noise_rate)
                    +'/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
                    +'/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           './test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) + str(Noise_rate) + '/'
                           + netname + '_' + str(participant_index) + '.ckpt')
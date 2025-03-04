import numpy as np
import torch
from utils.utils_dual import *
import os
from datasets.dataset_generic_dual import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_sum import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = 2
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    # log(prediction, label) updates the count of predictions and the number of correct predictions
    # (accuracy = correct_predictions / count)
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.tot_acc_min =np.Inf

    def __call__(self, epoch, val_loss, tot_acc, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss
        score2 = tot_acc

        if self.best_score is None:
            self.best_score = score2
            self.save_checkpoint(val_loss, tot_acc, model, ckpt_name)
        elif score2 <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score2
            self.save_checkpoint(val_loss, tot_acc, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, tot_acc, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Weighted accuracy increased ({self.tot_acc_min:.6f} --> {tot_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        self.tot_acc_min = tot_acc

def train(datasets, cur, args, pre_trained_model = None):
    """   
        train for a single fold
    """

    print('\nTraining Fold {}!'.format(cur), flush=True)
    # Path to the results dir where the results will be stored (e.g. /results/exp_code/i/)
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir,exist_ok=True)
    # Tensorboard logging
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    # Calculate the number of slides for each split
    num_train_slides = len(train_split.expanded_indices)
    num_val_slides = len(val_split.expanded_indices)
    num_test_slides = len(test_split.expanded_indices)

    print("Training on {} slides".format(num_train_slides))
    print("Validating on {} slides".format(num_val_slides))
    print("Testing on {} slides".format(num_test_slides))
    # print("Training on {} samples".format(len(train_split)))
    # print("Validating on {} samples".format(len(val_split)))
    # print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    ## LOSS FUNCTION SETTING
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    if pre_trained_model is not None:
        model = pre_trained_model
    else:
        print('\nInit Model...', end=' ')
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
        # Model type
        if args.model_type == 'clam' and args.subtyping:
            model_dict.update({'subtyping': True})
        # Model size
        if args.model_size is not None and args.model_type != 'mil':
            model_dict.update({"size_arg": args.model_size})
        # Model type 2: single branch, multiple branch
        if args.model_type in ['clam_sb', 'clam_mb']:
            if args.subtyping:
                model_dict.update({'subtyping': True})
            # B parameter for clustering
            if args.B > 0:
                model_dict.update({'k_sample': args.B})
            # Clustering Loss
            if args.inst_loss == 'svm':
                from topk.svm import SmoothTop1SVM
                instance_loss_fn = SmoothTop1SVM(n_classes = 2)
                if device.type == 'cuda':
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                instance_loss_fn = nn.CrossEntropyLoss()
            ## CREATION OF THE MODEL
            if args.model_type =='clam_sb':
                model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            elif args.model_type == 'clam_mb':
                model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                raise NotImplementedError
        
        else: # args.model_type == 'mil'
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)
        # Move the model to the DEVICE
        model.relocate()
        print('Done!')
        print_network(model)

    print('\nInit optimizer ...', end=' ')
    # Optimizer
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # Data loaders
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    # Early Stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 30, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!', flush=True)
    # loss_fn = nn.CrossEntropyLoss()
    # print('Setting Cross Entropy Loss...', flush=True)
    loss_fn = nn.BCELoss()
    print('Setting BCE Loss...', flush=True)


    for epoch in range(args.max_epochs):
        # If the model is CLAM with Instance Clustering
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            # Train loop for CLAM
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop, val_loss= validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop, val_loss = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
            
        scheduler.step(val_loss)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),  _use_new_zipfile_serialization=False)

    # Validation step
    _, val_error, val_auc, _ = summary_dual_stain(model, val_loader, args.n_classes)
    #_, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    
    # Test step
    results_dict, test_error, test_auc, acc_logger = summary_dual_stain(model, test_loader, args.n_classes)
    # results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  ##?
    # Accuracy Logger
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    focal = torchvision.ops.focal_loss

    print('\n')
    for batch_idx, (features_list, labels, coords_list, stains_list) in enumerate(loader):
        for slide_idx in range(len(features_list)):
            features = features_list[slide_idx]
            label = labels[slide_idx]
            coords = coords_list[slide_idx]
            stain = stains_list[slide_idx]
            #for slide_idx, (features, label, coords, stain) in zip(features_list, labels, coords_list, stains_list):
            # Data and labels to device
            features, label = features.to(device), label.to(device)
                # Model run: it returns the logits ([1, N_classes), the probability scores (softmax of logits),
                # Y_hat - that is the predicted label -, the patch features and the dictionary reporting some results,
                # such as the clustering loss, accuracy, etc.
            logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True)
                # accuracy update (predictions counter is incremented, correct predictions counter is incremented if
                # Y_hat is equal to label
            acc_logger.log(Y_hat, label)
                # classification loss
                #loss = loss_fn(Y_prob, label.float())
            loss = loss_fn(Y_prob, label.view_as(Y_prob).float())

                #labels = torch.nn.functional.one_hot(label, num_classes=2).float()

                #loss = focal.sigmoid_focal_loss(logits, labels, reduction = 'mean')

            loss_value = loss.item()
            # instance clustering loss is recovered from the instance_dict
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            # total loss is a weighted sum of the total loss with the instance loss; the weight is a parameter of the training
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

            # clustering accuracy update

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            error = calculate_error(Y_hat, label)
            train_error += error
            
            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            # Print progress every 20 slides
            if (batch_idx * len(features_list) + slide_idx + 1) % 20 == 0:
                print(
                    f"Batch {batch_idx}, Slide {slide_idx}, Loss: {loss_value:.4f}, "
                    f"Instance Loss: {instance_loss_value:.4f}, Weighted Loss: {total_loss.item():.4f}, "
                    f"Label: {label.item()}, Features Size: {features.size(0)}",
                    flush=True
                )



    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    #focal = torchvision.ops.focal_loss
    print('\n')
    for batch_idx, (features_list, labels, coords_list, stains_list) in enumerate(loader):
        for slide_idx in range(len(features_list)):
            features = features_list[slide_idx]
            label = labels[slide_idx]
            coords = coords_list[slide_idx]
            stain = stains_list[slide_idx]
         #for slide_idx, (features, label, coords, stain) in zip(features_list, labels, coords_list, stains_list):
            features, label = features.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(Y_prob, label.view_as(Y_prob).float())
            #loss = loss_fn(Y_prob, label.float())
            #labels = torch.nn.functional.one_hot(label, num_classes=1).float()
            #loss = focal.sigmoid_focal_loss(logits, labels, reduction = 'sum')
            loss_value = loss.item()
            train_loss += loss_value
            error = calculate_error(Y_hat, label)
            train_error += error

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()

            # Print every 20 slides
            if (batch_idx * len(features_list) + slide_idx + 1) % 20 == 0:
                print(
                    f"Batch {batch_idx}, Slide {slide_idx}, Loss: {loss.item():.4f}, "
                    f"Label: {label.item()}, Features Size: {features.size(0)}",
                    flush=True,
                )
           
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    """
    Validation logic for non-CLAM models with dual-stain support.
    Handles dual-stain batch structure and aggregates predictions at the donor level.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    donor_results = {}
    val_loss = 0.0
    val_error = 0.0
    print("\n=== Starting Validation ===")
    print(f"Number of batches in loader: {len(loader)}")

    with torch.no_grad():
        for batch_idx, (features_list, labels, coords_list, stains_list) in enumerate(loader):
            print(f"\nProcessing batch {batch_idx + 1}/{len(loader)}")
            print(f"Batch size: {len(features_list)}")
            for slide_idx in range(len(features_list)):
                print(f"  Processing slide {slide_idx + 1}/{len(features_list)} in batch {batch_idx + 1}")
                # Extract slide-specific data
                features = features_list[slide_idx]
                label = labels[slide_idx]
                coords = coords_list[slide_idx]
                stain = stains_list[slide_idx]

                print(f"    Slide stain: {stain}, Label: {label}")

                # Move features and labels to device
                features = features.to(device)
                if not isinstance(label, torch.Tensor):  # Ensure label is a tensor
                    label = torch.tensor([label], dtype=torch.float32).to(device)
                else:
                    label = label.to(device).float()
                print(f"    Features shape: {features.shape}, Label shape: {label.shape}")                
                # Forward pass
                logits, Y_prob, Y_hat, _, _ = model(features)
                print(f"    Predicted probabilities: {Y_prob.cpu().numpy()}, Predicted label: {Y_hat.cpu().item()}")                
                
                label = label.view_as(Y_prob)

                # Map slide_idx to base_idx using expanded_indices
                expanded_index = loader.dataset.expanded_indices[batch_idx * len(features_list) + slide_idx]
                base_idx, stain = expanded_index # Unpack base index in slide_data
                # Fetch donor_id from slide_data
                donor_id = loader.dataset.slide_data.iloc[base_idx]["case_id"]
                print(f"Batch {batch_idx}, Slide {slide_idx}, Expanded index: {expanded_index}")
                print(f"    Donor ID: {donor_id}, Base index: {base_idx}, Stain: {stain}")
                
                prob = Y_prob.cpu().numpy()
                label_value = label.item()  # Use this for logging, not for computation

                if donor_id not in donor_results:
                    donor_results[donor_id] = {'probs': [], 'label': label_value, 'preds': []}

                donor_results[donor_id]['probs'].append(prob)
                donor_results[donor_id]['preds'].append(Y_hat.cpu().item())

                # Log loss and error
                loss = loss_fn(Y_prob, label)
                val_loss += loss.item()
                print(f"    Loss: {loss.item()}")

                error = calculate_error(Y_hat, label)
                val_error += error
                print(f"    Error: {error}")

    print("\n=== Aggregating Results ===")
    # Aggregate donor-level predictions
    all_labels, all_probs, all_preds = [], [], []
    for donor_id, results in donor_results.items():
        max_prob = np.max(results['probs'], axis=0)  # Aggregate via max
        pred = np.argmax(max_prob)  # Final prediction
        aggregated_result = {
            'prob': max_prob,
            'label': results['label'],
            'pred': pred,
        }

        print(f"  Donor ID: {donor_id}, Aggregated probabilities: {max_prob}, Predicted label: {pred}, True label: {results['label']}")

        all_probs.append(max_prob)
        all_labels.append(results['label'])
        all_preds.append(pred)

        acc_logger.log(pred, results['label'])


    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Check for single-class issue
    unique_classes = np.unique(all_labels)
    if len(unique_classes) < 2:
        print("Warning: Only one class present in validation labels. Skipping ROC AUC calculation.")
        auc = float('nan')
    else:
        if n_classes == 2:
            if all_probs.shape[1] == 1:
                all_probs = np.hstack([1 - all_probs, all_probs])
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                label_binarize(all_labels, classes=range(n_classes)), all_probs, multi_class="ovr"
            )
    # Calculate metrics
    val_error /= len(loader.dataset)
    val_loss /= len(loader.dataset)
    if n_classes == 2:
        # Ensure binary classification handles single-column probabilities
        if all_probs.shape[1] == 1:
            all_probs = np.hstack([1 - all_probs, all_probs])  # Add probabilities for the negative class
        auc = roc_auc_score(all_labels, all_probs[:, 1])  # Use probabilities for the positive class
    else:
        auc = roc_auc_score(
            label_binarize(all_labels, classes=range(n_classes)), all_probs, multi_class="ovr"
        )

    # Log results
    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", auc, epoch)
        writer.add_scalar("val/error", val_error, epoch)

    print(f"\nVal Set: val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}")
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc {acc}, correct {correct}/{count}")

        if writer and acc is not None:
            writer.add_scalar(f"val/class_{i}_acc", acc, epoch)

    # Early stopping
    if early_stopping:
        assert results_dir
        early_stopping(
            epoch, val_loss, np.mean(all_preds == all_labels), model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss


   
# def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     # loader.dataset.update_mode(True)
#     val_loss = 0.
#     val_error = 0.
    
#     prob = np.zeros((len(loader), n_classes))
#     labels = np.zeros(len(loader))

#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(loader):
#             data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

#             logits, Y_prob, Y_hat, _, _ = model(data)

#             acc_logger.log(Y_hat, label)
            
#             loss = loss_fn(Y_prob, label.float())

#             prob[batch_idx] = Y_prob.cpu().numpy()
#             labels[batch_idx] = label.item()
            
#             val_loss += loss.item()
#             error = calculate_error(Y_hat, label)
#             val_error += error
            

#     val_error /= len(loader)
#     val_loss /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(labels, prob[:, 1])
    
#     else:
#        # print(labels, prob)
#         auc = roc_auc_score(labels, prob, multi_class='ovr')
#         print('ACC LOGGER ', acc_logger)
    
    
#     if writer:
#         writer.add_scalar('val/loss', val_loss, epoch)
#         writer.add_scalar('val/auc', auc, epoch)
#         writer.add_scalar('val/error', val_error, epoch)

#     print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer and acc is not None:
#             writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

#     total_correct = 0
#     total_count = 0
#     for i in range(n_classes):
#         _, correct, count = acc_logger.get_summary(i)
#         total_correct+=correct
#         total_count +=count

#     tot_acc = (total_correct/total_count)
    
#     if early_stopping:
#         assert results_dir
#         early_stopping(epoch, val_loss, tot_acc, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             return True, val_loss

#     return False, val_loss

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    """
    Validation logic for CLAM models with dual-stain support.
    Handles dual-stain batch structure and aggregates predictions at the donor level.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    donor_results = {}
    val_loss = 0.0
    val_error = 0.0
    inst_count = 0

    with torch.no_grad():
        for batch_idx, (features_list, labels, coords_list, stains_list) in enumerate(loader):
            for slide_idx in range(len(features_list)):
                # Extract slide-specific data
                features = features_list[slide_idx]
                label = labels[slide_idx]
                coords = coords_list[slide_idx]
                stain = stains_list[slide_idx]

                # Move features and labels to device
                features = features.to(device)
                # Forward pass with instance evaluation
                logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True)                
                
                label = torch.tensor([label], device=device, dtype=torch.float32).view_as(Y_prob)

                # Donor-level aggregation
                donor_id = loader.dataset.slide_data.iloc[batch_idx]["case_id"]
                prob = Y_prob.cpu().numpy()
                label = label.item()

                if donor_id not in donor_results:
                    donor_results[donor_id] = {'probs': [], 'label': label, 'preds': []}

                donor_results[donor_id]['probs'].append(prob)
                donor_results[donor_id]['preds'].append(Y_hat.cpu().item())

                # Log loss and instance metrics
                loss = loss_fn(Y_prob, label)
                val_loss += loss.item()

                instance_loss = instance_dict['instance_loss']
                inst_count += 1
                inst_logger.log_batch(instance_dict['inst_preds'], instance_dict['inst_labels'])

                error = calculate_error(Y_hat, torch.tensor([label]))
                val_error += error

    # Aggregate donor-level predictions
    all_labels, all_probs, all_preds = [], [], []
    for donor_id, results in donor_results.items():
        max_prob = np.max(results['probs'], axis=0)  # Aggregate via max
        pred = np.argmax(max_prob)  # Final prediction

        all_probs.append(max_prob)
        all_labels.append(results['label'])
        all_preds.append(pred)

        acc_logger.log(pred, results['label'])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate metrics
    val_error /= len(loader.dataset)
    val_loss /= len(loader.dataset)
    auc = (
        roc_auc_score(all_labels, all_probs[:, 1])
        if n_classes == 2
        else roc_auc_score(
            label_binarize(all_labels, classes=range(n_classes)), all_probs, multi_class="ovr"
        )
    )

    # Log results
    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", auc, epoch)
        writer.add_scalar("val/error", val_error, epoch)

    print(f"\nVal Set: val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}")
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc {acc}, correct {correct}/{count}")

        if writer and acc is not None:
            writer.add_scalar(f"val/class_{i}_acc", acc, epoch)

    # Early stopping
    if early_stopping:
        assert results_dir
        early_stopping(
            epoch, val_loss, np.mean(all_preds == all_labels), model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss



# def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     inst_logger = Accuracy_Logger(n_classes=n_classes)
#     val_loss = 0.
#     val_error = 0.

#     val_inst_loss = 0.
#     val_inst_acc = 0.
#     inst_count=0
    
#     prob = np.zeros((len(loader), n_classes))
#     labels = np.zeros(len(loader))
#     sample_size = model.k_sample
#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(loader):
#             data, label = data.to(device), label.to(device)      
#             logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
#             acc_logger.log(Y_hat, label)
            
#             loss = loss_fn(Y_prob, label.float())

#             val_loss += loss.item()

#             instance_loss = instance_dict['instance_loss']
            
#             inst_count+=1
#             instance_loss_value = instance_loss.item()
#             val_inst_loss += instance_loss_value

#             inst_preds = instance_dict['inst_preds']
#             inst_labels = instance_dict['inst_labels']
#             inst_logger.log_batch(inst_preds, inst_labels)

#             prob[batch_idx] = Y_prob.cpu().numpy()
#             labels[batch_idx] = label.item()
            
#             error = calculate_error(Y_hat, label)
#             val_error += error

#     val_error /= len(loader)
#     val_loss /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(labels, prob[:, 1])
#         aucs = []
#     else:
#         aucs = []
#         binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
#         for class_idx in range(n_classes):
#             if class_idx in labels:
#                 fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
#                 aucs.append(calc_auc(fpr, tpr))
#             else:
#                 aucs.append(float('nan'))

#         auc = np.nanmean(np.array(aucs))

#     print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
#     if inst_count > 0:
#         val_inst_loss /= inst_count
#         for i in range(2):
#             acc, correct, count = inst_logger.get_summary(i)
#             print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
#     if writer:
#         writer.add_scalar('val/loss', val_loss, epoch)
#         writer.add_scalar('val/auc', auc, epoch)
#         writer.add_scalar('val/error', val_error, epoch)
#         writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
#         if writer and acc is not None:
#             writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

#     total_correct = 0
#     total_count = 0
#     for i in range(n_classes):
#         _, correct, count = acc_logger.get_summary(i)
#         total_correct+=correct
#         total_count +=count

#     tot_acc = (total_correct/total_count)
     

#     if early_stopping:
#         assert results_dir
#         early_stopping(epoch, val_loss, tot_acc, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             return True, val_loss

#     return False, val_loss

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

def summary_dual_stain(model, loader, n_classes):
    """
    Validation/test logic to aggregate predictions for donors with multiple stains.
    Returns:
        results_dict: A dictionary containing probabilities, labels, and predictions for each donor.
        error: Average classification error across donors.
        auc: ROC AUC score across donors.
        acc_logger: Accuracy logger containing class-specific accuracy statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()

    donor_results = {}

    with torch.no_grad():
        for batch_idx, (features_list, labels, coords_list, stains_list) in enumerate(loader):
            for slide_idx, (features, label, coords, stain) in enumerate(zip(features_list, labels, coords_list, stains_list)):
                # Move features and labels to device
                features = features.to(device)
                label = label.to(device)

                # Model inference
                logits, Y_prob, Y_hat, _, _ = model(features)

                # Resolve donor ID using expanded_indices
                expanded_index = loader.dataset.expanded_indices[batch_idx * len(features_list) + slide_idx]
                base_idx = expanded_index[0]
                donor_id = loader.dataset.slide_data.loc[base_idx, "case_id"]

                prob = Y_prob.cpu().numpy()  # Move probabilities to CPU
                label_value = label.cpu().item()  # Move label to CPU for storage
                print(f"Donor ID: {donor_id}, Label: {label_value}, Prob: {prob}")

                if donor_id not in donor_results:
                    donor_results[donor_id] = {
                        'probs': [],
                        'label': label_value,  # Store the CPU value
                        'preds': []
                    }

                donor_results[donor_id]['probs'].append(prob)
                donor_results[donor_id]['preds'].append(Y_hat.cpu().item())  # Ensure Y_hat is on CPU

    print(f"Donor results: {donor_results}")        
    # Aggregate predictions for each donor
    aggregated_results = {}
    all_labels = []
    all_preds = []
    all_probs = []

    for donor_id, results in donor_results.items():
        # Find the index of the maximum probability
        max_prob_idx = np.argmax([np.max(prob) for prob in results['probs']])
        max_prob = results['probs'][max_prob_idx]  # Probability from the slide with max prob
        pred = results['preds'][max_prob_idx]  # Prediction corresponding to the max_prob

        print(f"Processing donor {donor_id}: max_prob={max_prob}, pred={pred}")

        aggregated_results[donor_id] = {
            'prob': max_prob,
            'label': results['label'],  # Already on CPU
            'pred': pred,
        }

        all_probs.append(max_prob)
        all_labels.append(results['label'])  # Append the CPU-stored label
        all_preds.append(pred)

        acc_logger.log(pred, results['label'])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)  # Ensure all labels are on CPU
    all_preds = np.array(all_preds)
    # Debugging prints for shapes
    print(f"Shape of all_probs: {all_probs.shape}")
    print(f"Shape of all_labels: {all_labels.shape}")
    print(f"Sample of all_probs: {all_probs[:5]}")

    # Handle single-class probabilities for binary classification
    if n_classes == 2 and all_probs.shape[1] == 1:
        all_probs = np.hstack([1 - all_probs, all_probs])
        print("Adjusted probabilities for binary classification with single-column output.")

    # Calculate AUC
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=range(n_classes))
        auc = roc_auc_score(binary_labels, all_probs, multi_class='ovr')

    # Calculate classification error
    error = np.mean(all_preds != all_labels)

    return aggregated_results, error, auc, acc_logger



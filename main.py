import time
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CombineGraph, forward, trans_to_cuda, trans_to_cpu
from utils import Data, split_validation, handle_adj


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='diginetica/yoochoose1_64/Nowplaying/Tmall')
parser.add_argument('--mode', default='train', help='train/test')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion for validation')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--model_path', type=str, default='latest_model.pth', help='Path to save/load the model')

opt = parser.parse_args()


def evaluate(model, data_loader):
    """Evaluates the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    hit, mrr = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            targets, scores = forward(model, data)
            targets = trans_to_cuda(targets).long()
            loss = model.loss_function(scores, targets - 1)
            total_loss += loss.item() * targets.size(0)
            
            # Calculate metrics
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets_np = targets.cpu().numpy()
            
            for score, target in zip(sub_scores, targets_np):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    avg_loss = total_loss / len(data_loader.dataset)
    mean_hit = np.mean(hit) * 100
    mean_mrr = np.mean(mrr) * 100
    
    return avg_loss, mean_hit, mean_mrr


def plot_metrics(history):
    """Plots training and validation metrics."""
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['valid_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Metrics (Hit@20 and MRR@20)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_hit'], 'bs-', label='Training Recall@20')
    plt.plot(epochs, history['valid_hit'], 'rs-', label='Validation Recall@20')
    plt.plot(epochs, history['train_mrr'], 'gd-', label='Training MRR@20')
    plt.plot(epochs, history['valid_mrr'], 'yd-', label='Validation MRR@20')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_validation_plots.png')
    print("\nPlots saved to training_validation_plots.png")
    plt.show()


def main():
    init_seed(2020)

    # --- Dataset-specific configurations ---
    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        num_node = 37484
    elif opt.dataset == 'yoochoose1_64_5':
        num_node = 21779
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    else:
        num_node = 310

    # --- Data Loading ---
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data_raw = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    
    train_data, valid_data = split_validation(train_data, opt.valid_portion)
    
    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    
    train_data = Data(train_data)
    valid_data = Data(valid_data)
    test_data = Data(test_data_raw)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, num_workers=4, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print(opt)

    if opt.mode == 'train':
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        
        history = {'train_loss': [], 'valid_loss': [], 'train_hit': [], 'valid_hit': [], 'train_mrr': [], 'valid_mrr': []}

        for epoch in range(opt.epoch):
            print(f'------------------- Epoch: {epoch+1}/{opt.epoch} -------------------')
            
            # --- Training Phase ---
            model.train()
            total_epoch_loss = 0.0
            for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                model.optimizer.zero_grad()
                targets, scores = forward(model, data)
                targets = trans_to_cuda(targets).long()
                loss = model.loss_function(scores, targets - 1)
                loss.backward()
                model.optimizer.step()
                total_epoch_loss += loss.item() * targets.size(0)
            model.scheduler.step()
            
            # --- Evaluation Phase ---
            print(f"\n--- Evaluating Epoch {epoch+1} ---")
            # Note: Evaluating on the full training set can be slow. For faster checks, consider using a subset.
            train_loss, train_hit, train_mrr = evaluate(model, train_loader)
            valid_loss, valid_hit, valid_mrr = evaluate(model, valid_loader)

            # Store history for plotting
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_hit'].append(train_hit)
            history['valid_hit'].append(valid_hit)
            history['train_mrr'].append(train_mrr)
            history['valid_mrr'].append(valid_mrr)

            print(f"Train Results: Loss: {train_loss:.4f}, Recall@20: {train_hit:.4f}%, MRR@20: {train_mrr:.4f}%")
            print(f"Valid Results: Loss: {valid_loss:.4f}, Recall@20: {valid_hit:.4f}%, MRR@20: {valid_mrr:.4f}%")

            # --- Early Stopping and Best Model Tracking ---
            flag = 0
            if valid_hit >= best_result[0]:
                best_result[0] = valid_hit
                best_epoch[0] = epoch + 1
                flag = 1
            if valid_mrr >= best_result[1]:
                best_result[1] = valid_mrr
                best_epoch[1] = epoch + 1
                flag = 1
            
            print('Best Validation Result:')
            print(f'\tRecall@20: {best_result[0]:.4f}%, MRR@20: {best_result[1]:.4f}%, Epoch: {best_epoch[0]},{best_epoch[1]}')
            
            bad_counter += 1 - flag
            if bad_counter >= opt.patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in {opt.patience} epochs.")
                break
        
        print('-------------------------------------------------------')
        print("Training finished.")
        
        # --- Save the final model ---
        print(f"Saving the model from the last epoch to {opt.model_path}")
        torch.save(model.state_dict(), opt.model_path)
        
        end = time.time()
        print(f"Total Run time: {(end - start) / 60:.2f} minutes")

        # --- Plotting ---
        plot_metrics(history)

        # --- Final Test Evaluation ---
        print("\n--- Running final evaluation on the Test Set ---")
        test_loss, test_hit, test_mrr = evaluate(model, test_loader)
        print(f"Final Test Results: Loss: {test_loss:.4f}, Recall@20: {test_hit:.4f}%, MRR@20: {test_mrr:.4f}%")

    elif opt.mode == 'test':
        print("\n--- Test Mode: Loading model and evaluating on the test set ---")
        try:
            model.load_state_dict(torch.load(opt.model_path))
            print(f"Model loaded successfully from {opt.model_path}")
            
            test_loss, test_hit, test_mrr = evaluate(model, test_loader)
            print(f"Final Test Results: Loss: {test_loss:.4f}, Recall@20: {test_hit:.4f}%, MRR@20: {test_mrr:.4f}%")

        except FileNotFoundError:
            print(f"Error: Model file not found at {opt.model_path}. Please train a model first using --mode train.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

if __name__ == '__main__':
    main()
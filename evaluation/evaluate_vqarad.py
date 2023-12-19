import _pickle as cPickle
import argparse
import json
import os
import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils.data_vqarad import _load_dataset
from data_utils.data_vqarad import _load_dataset, create_image_to_question_dict, VQARad
from net.model import ModelWrapper,Model

warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Finetune on VQARAD")

    parser.add_argument('--run_name', type=str, required=False, default="debug", help="run name for wandb")
    parser.add_argument('--data_dir', type=str, required=False, default="data/vqarad", help="path for data")
    parser.add_argument('--model_dir', type=str, required=False, default="",
                        help="path to load weights")
    parser.add_argument('--save_dir', type=str, required=False, default="checkpoints", help="path to save weights")
    parser.add_argument('--question_type', type=str, required=False, default=None, help="choose specific category if you want")
    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")
    parser.add_argument('--clip', action='store_true', default=False, help="clip the gradients or not")

    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")

    parser.add_argument('--progressive', action='store_true', default=False, help="use progressive answering of questions")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=12, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--max_answer_len', type=int, required=False, default=29, help="padding length for free-text answers")
    parser.add_argument('--batch_size', type=int, required=False, default=2, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")
    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")
    parser.add_argument('--smoothing', type=float, required=False, default=None, help="label smoothing")

    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=30, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--n_block', type=int, required=False, default=10, help="number of prototype block")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=8, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of fusion layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")
    parser.add_argument('--scaling', type=int, required=False, default=0.2, help="scaling in hopfield")
    parser.add_argument('--quantity', type=int, required=False, default=2, help="Quantity in hopfield pooling")
    parser.add_argument('--classifier_hopfield', type=int, required=False, default=0.5, help="Quantity in hopfield pooling")
    parser.add_argument('--slot_size', type=int, required=False, default=96, help="STM slot size")
    parser.add_argument('--rel_size', type=int, required=False, default=96*2, help="STM rel_size")
    parser.add_argument('--mlp_size', type=int, required=False, default=256, help="STM mlp")
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.4, help="how often should image be dropped")
    ###mca
    parser.add_argument('--mca_hidden_size', type=int, required=False, default=768, help="MCA hidden size")
    parser.add_argument('--mca_layer', type=int, required=False, default=3, help="MCA hidden size")
    parser.add_argument('--mca_heads', type=int, required=False, default=8, help="MCA number head")
    parser.add_argument('--mca_ff', type=int, required=False, default=768, help="MCA feedforward size")
    parser.add_argument('--mca_dropout', type=float, required=False, default=0.2, help="MCA dropout rate")
    
    args = parser.parse_args()

    # create directory for saving params
    args.mca_hidden_size_heads = int(args.mca_hidden_size / args.mca_heads)
    if not os.path.exists(f'{args.save_dir}/{args.run_name}'):
        os.makedirs(f'{args.save_dir}/{args.run_name}')
    with open(os.path.join(args.save_dir, f'{args.run_name}/commandline_args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    pl.seed_everything(args.seed, workers=True)

    train_df = _load_dataset(args.data_dir, 'train')
    val_df = _load_dataset(args.data_dir, 'test')

    img_to_q_dict_train, img_to_q_dict_all = create_image_to_question_dict(train_df, val_df)

    args.num_classes = 495

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #m = ModelWrapper(args)
    model = Model(args)
    model = model.cuda()
    # use torchinfo to see model architecture and trainable parameters
    from torchinfo import summary

    summary(model)

    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu'))['state_dict'])

    img_tfm = model.image_encoder.img_tfm
    norm_tfm = model.image_encoder.norm_tfm
    resize_size = model.image_encoder.resize_size

    test_tfm = transforms.Compose([img_tfm, norm_tfm])
    valdataset = VQARad(val_df, tfm=test_tfm, args=args)
    valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # preds = trainer.predict(model, valloader, return_predictions=True)
    # given the valloader and the predictions pred, compute the accuracy for each batch and create a list of the wrong examples
    # load trainval_label2ans.pkl
    with open('data/vqarad/trainval_label2ans.pkl', 'rb') as f:
        label2ans = cPickle.load(f)
    correct = 0
    total = 0
    model.eval()
    for i, batch in tqdm(enumerate(valloader)):
        #print(batch)
        img, question_token, q_attention_mask, target,answer_type  = batch
        target = target.cuda()
        question_token = question_token.cuda()
        q_attention_mask = q_attention_mask.cuda()
        img = img.cuda()
        # convert all to tensor
        out= model(img, question_token, q_attention_mask)
        #print(out.shape)
        logits = out
        pred = logits.softmax(1).argmax(1).detach()
        #text_pred = label2ans[pred.item()]

        correct += (pred == target).sum().item()
        total += 1

    print(correct)
    print(total)
    print(f'Accuracy: {correct / total * 100.:.2f}')

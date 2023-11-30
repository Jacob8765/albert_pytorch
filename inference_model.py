from model.modeling_albert import AlbertForSequenceClassification
from model import tokenization_albert
import os
import torch
from processors.glue import glue_convert_examples_to_features as convert_examples_to_features
from processors.utils import InputExample
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import tqdm

# STR1 = "This is a great product, I'm really impressed :)"
# STR2 = "This is a bad product, I'm really disappointed :("
# STR3 = "This is a product, I'm really disappointed :("

# data = [
# {
#     'sentence_1': STR1, 
#     "test col": 1
# }, 
# {
#     'sentence_1': STR2, 
#     "test col": 1
# }, 
# {
#     'sentence_1': STR3,
#     "test col": 1
# }]

ex_dataframe = pd.DataFrame(data)

def inference_model(df=ex_dataframe, col_name="sentence_1"):
    vocab_file = "./prev_trained_model/albert_base_v2/30k-clean.vocab"
    do_lower_case = True
    spm_model_file = "./prev_trained_model/albert_base_v2/30k-clean.model"

    output_dir = os.path.join(os.getcwd(), "outputs/sst-2_output")
    checkpoint_path = os.path.join(output_dir, "albert", "checkpoint-8420")

    model = AlbertForSequenceClassification.from_pretrained(checkpoint_path)
    model.to("mps")
    model.eval()

    tokenizer = tokenization_albert.FullTokenizer(vocab_file=vocab_file,
                                                      do_lower_case=do_lower_case,
                                                      spm_model_file=spm_model_file)

    col_values = df[col_name].values.tolist()

    raw_input = [InputExample(guid=f"test-{i}", text_a=text, text_b=None, label="0") for i, text in enumerate(col_values)]

    features = convert_examples_to_features(examples=raw_input, label_list=["0", "1"], max_seq_length=128, tokenizer=tokenizer, output_mode="classification")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)

    eval_dataloader = DataLoader(dataset, batch_size=32)

    results = []

    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Inference Progress"):
        #move the batch to mps
        batch = tuple(t.to("mps") for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]

            outputs = model(**inputs)
            logits = outputs[1]

            #return the sentiment (probability of the positive class))
            logits_sigmoid = torch.sigmoid(logits)
            logits_sigmoid = logits_sigmoid.detach().cpu().numpy()[:, 1]

            predictions = logits_sigmoid.tolist()
            results.extend(predictions)

    df["review_sentiment"] = results
    return df

if __name__ == "__main__":
    inference_model()
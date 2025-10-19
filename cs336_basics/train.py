import numpy as np
import module
import tokenizer
from pathlib import Path
import time
import torch

FIXTURES_PATH = Path(__file__).parent / "../tests" / "fixtures"

def main():
    d_model = 512
    num_heads = 16
    d_ff = 1344
    context_length = 256
    rope_theta = 10000.0
    num_layers = 4
    vocab_size = 10000
    max_l2_norm = 1e-2
    warmup_iters = 500
    total_iters = 5000
    lr_min = 1e-5
    lr_max = 1e-3
    batch_size = 32
    
    device = torch.device("mps")
    torch._dynamo.disable()
    torch.set_default_device(device)

    model = module.TransformerLM(d_model, num_heads, d_ff, context_length, rope_theta, num_layers, vocab_size)
    model = torch.compile(model, backend="aot_eager")

    print("Default device:", torch.get_default_device())

    optimizer = module.AdamW(model.parameters())

    loss_fn = module.CrossEntropyLoss()

    data = np.memmap("/Users/mazijie/Documents/assignment1-basics/TinyStories_train.bin", dtype=np.uint16, mode="r")
    # valid_data = np.memmap("/Users/mazijie/Documents/assignment1-basics/TinyStories_valid.bin", dtype=np.uint16, mode="r")

    iter = 0
    if Path("checkpoint.pth").exists():
        print("Loading checkpoint...")
        iter = module.load_checkpoint("checkpoint.pth", model, optimizer)
    time_start = time.time()
    for i in range(iter, total_iters):
        inputs, label = module.get_batch(data, batch_size, context_length, device)
        lr = module.get_lr_cosine_schedule(
            i,
            max_learning_rate = lr_max,
            min_learning_rate = lr_min,
            warmup_iters = warmup_iters,
            cosine_cycle_iters = total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        result = model(inputs)
        loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
        loss.backward()
        module.gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

        print(f"Iteration {i}, Loss: {loss.item()}")
        
        # if i % 100 == 0:
        #     model.eval()
        #     valid_loss = 0
        #     valid_count = 0
        #     valid_batch_size = 0
        #     for inputs, label in module.get_memmap_batch(valid_data, batch_size, context_length, device):
        #         result = model(inputs)
        #         loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
        #         valid_loss += loss.item() * label.shape[0] * label.shape[1]
        #         valid_count += label.shape[0] * label.shape[1]
        #         valid_batch_size += 1
        #         if valid_batch_size >= 10:
        #             break
        #     print(f"Validation Loss: {valid_loss / valid_count}")
        #     model.train()
    print(f"Time taken: {time.time() - time_start} seconds")

    module.save_checkpoint(model, optimizer, total_iters, "checkpoint.pth")
    
def train_bpe_translate():
    bpe_tokenizer = tokenizer.Tokenizer.from_files(FIXTURES_PATH / "TinyStoriesV2-GPT4-vocab.json", FIXTURES_PATH / "TinyStoriesV2-GPT4-merges.txt")
    print("begin to translate dataset")
    dataset_translate = tokenizer.DataSetTranslate(bpe_tokenizer)

    train_file_path = FIXTURES_PATH / "../../data/TinyStoriesV2-GPT4-train.txt"
    train_output_path = FIXTURES_PATH / "TinyStoriesV2-GPT4-train.bin"
    dataset_translate.call(train_file_path, train_output_path)

    valid_file_path = FIXTURES_PATH / "../../data/TinyStoriesV2-GPT4-valid.txt"
    valid_output_path = FIXTURES_PATH / "TinyStoriesV2-GPT4-valid.bin"
    dataset_translate.call(valid_file_path, valid_output_path)

if __name__ == "__main__":
    # train_bpe_translate()
    main()
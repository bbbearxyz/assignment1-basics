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
    clip_grad_norm = 1.0
    warmup_iters = 500
    total_iters = 5000
    lr_min = 1e-4
    lr_max = 2e-3
    batch_size = 32
    
    # type = "evaluate"
    type = "train"
    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_float32_matmul_precision('high')

    model = module.TransformerLM(d_model, num_heads, d_ff, context_length, rope_theta, num_layers, vocab_size)
    model = torch.compile(model)
    print("Default device:", torch.get_default_device())

    optimizer = module.AdamW(model.parameters())
    
    loss_fn = module.CrossEntropyLoss()

    data = np.memmap("./TinyStoriesV2-GPT4-train.bin", dtype=np.uint16, mode="r")
    valid_data = np.memmap("./TinyStoriesV2-GPT4-valid.bin", dtype=np.uint16, mode="r")
    
    iter = 0
    if Path("checkpoint.pth").exists():
        print("Loading checkpoint...")
        iter = module.load_checkpoint("checkpoint.pth", model, optimizer)

    if type == "evaluate":
        valid_loss = evaluate(model, loss_fn, valid_data, batch_size, context_length, device, vocab_size, max_count=100)
        print(f"Validation Loss: {valid_loss}")
        return

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
        module.gradient_clipping(model.parameters(), clip_grad_norm)
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
        
        if i % 100 == 0:
            inputs, label = module.get_batch(valid_data, batch_size, context_length, device)
            model.eval()
            result = model(inputs)
            loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
            model.train()
            print(f"Validation Loss: {loss.item()}")


    module.save_checkpoint(model, optimizer, total_iters, "checkpoint.pth")

def evaluate(model, loss_fn, data, batch_size, context_length, device, vocab_size, max_count):
    model.eval()
    with torch.no_grad():
        loss = 0
        count = 0
        for inputs, label in module.get_memmap_batch(data, batch_size, context_length, device):
            result = model(inputs)
            loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
            loss += loss.item() * label.shape[0] * label.shape[1]
    model.train()
    return loss / (batch_size * context_length)

def evaluate(model, loss_fn, data, batch_size, context_length, device, vocab_size, max_count):
    model.eval()
    with torch.no_grad():
        loss = 0
        count = 0
        for inputs, label in module.get_memmap_batch(data, batch_size, context_length, device):
            result = model(inputs)
            loss = loss_fn(result.view(-1, vocab_size), label.view(-1))
            loss += loss.item() * label.shape[0] * label.shape[1]
    model.train()
    return loss / (batch_size * context_length)

def train_bpe_translate():
    bpe_tokenizer = tokenizer.Tokenizer.from_files("./TinyStoriesV2-GPT4-vocab.json", "./TinyStoriesV2-GPT4-merges.txt")
    print("begin to translate dataset")
    dataset_translate = tokenizer.DataSetTranslate(bpe_tokenizer)

    time_start = time.time()
    train_file_path = "./data/TinyStoriesV2-GPT4-train.txt"
    train_output_path = "./TinyStoriesV2-GPT4-train.bin"
    dataset_translate.call(train_file_path, train_output_path)
    print(f"Time taken: {time.time() - time_start} seconds")

    time_start = time.time()
    valid_file_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    valid_output_path = "./TinyStoriesV2-GPT4-valid.bin"
    dataset_translate.call(valid_file_path, valid_output_path)
    print(f"Time taken: {time.time() - time_start} seconds")

def train_bpe():
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab, merges = tokenizer.train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    bpe_tokenizer = tokenizer.Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    bpe_tokenizer.write_files("./TinyStoriesV2-GPT4-vocab.json", "./TinyStoriesV2-GPT4-merges.txt")

if __name__ == "__main__":
    # train_bpe()
    # train_bpe_translate()
    main()
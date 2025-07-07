# Token Counter

A Python script that counts tokens in various file formats using OpenAI's Tiktoken library or HuggingFace Transformers.

## Features

- **Multiple tokenizer support**: OpenAI Tiktoken and HuggingFace AutoTokenizer (with use_fast=True)
- **Multiple file format support**: JSON, JSONL, TXT, MD, CSV, XLSX
- **Batch processing**: Process multiple directories from a paths file
- **Flexible file filtering**: Specify which file types to process
- **Detailed reporting**: CSV output with summary and optional detailed breakdown
- **Progress tracking**: Visual progress bar during processing
- **Error handling**: Robust error handling with detailed error reporting

## Installation

This project uses UV as the package manager. First, install UV if you haven't already:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Install dependencies
uv sync
```

## Usage

### Basic Usage

1. **Create a directory paths file** (default: `directory_paths.txt`):
   ```
   # Directory paths to process (one per line)
   /path/to/your/data/folder1
   /path/to/your/data/folder2
   ./relative/path/to/data
   ```

2. **Run the script**:
   ```bash
   # Process all supported file types in directories listed in directory_paths.txt
   uv run python token_counter.py
   
   # Or use the installed command
   uv run token-count
   ```

### Advanced Usage

```bash
# Specify custom paths file and file types
uv run python token_counter.py --paths-file my_paths.txt --file-types json,txt,csv

# Use different output file and tiktoken model
uv run python token_counter.py --output results.csv --model gpt-3.5-turbo

# Use HuggingFace tokenizer with BERT
uv run python token_counter.py --tokenizer huggingface --hf-model bert-base-uncased

# Use HuggingFace tokenizer with GPT-2
uv run python token_counter.py --tokenizer huggingface --hf-model gpt2

# Use HuggingFace tokenizer with a specific model and detailed output
uv run python token_counter.py --tokenizer huggingface --hf-model microsoft/DialoGPT-large --detailed

# Generate detailed file-by-file breakdown with tiktoken
uv run python token_counter.py --detailed

# Process only specific file types with HuggingFace tokenizer
uv run python token_counter.py --tokenizer huggingface --hf-model roberta-base --file-types jsonl,md
```

### Command Line Options

- `--paths-file, -p`: Text file containing directory paths (default: `directory_paths.txt`)
- `--file-types, -t`: Comma-separated list of file types to process (default: all supported types)
- `--output, -o`: Output CSV file name (default: `token_counts.csv`)
- `--tokenizer, --tokenizer-type`: Tokenizer type to use: `tiktoken` or `huggingface` (default: `tiktoken`)
- `--model, -m`: Model name for tiktoken (e.g., gpt-4, gpt-3.5-turbo) (default: `gpt-4`)
- `--hf-model`: HuggingFace model name when using `--tokenizer huggingface` (e.g., bert-base-uncased, gpt2)
- `--detailed, -d`: Include detailed file-by-file breakdown in output

### Supported Tokenizers

#### OpenAI Tiktoken
- **gpt-4**: Latest GPT-4 model tokenizer
- **gpt-3.5-turbo**: GPT-3.5 Turbo tokenizer
- **text-davinci-003**: Legacy GPT-3 model tokenizer
- **cl100k_base**: Base encoding for GPT-4 models
- **p50k_base**: Encoding for Codex models
- **r50k_base**: Encoding for GPT-3 models

#### HuggingFace AutoTokenizer (use_fast=True)
- **BERT models**: `bert-base-uncased`, `bert-large-cased`, etc.
- **GPT-2 models**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **RoBERTa models**: `roberta-base`, `roberta-large`
- **DistilBERT models**: `distilbert-base-uncased`
- **T5 models**: `t5-small`, `t5-base`, `t5-large`
- **And many more**: Any model available on HuggingFace Hub with AutoTokenizer support

### Supported File Types

- **JSON** (`.json`): Standard JSON files
- **JSONL** (`.jsonl`): JSON Lines format
- **Text** (`.txt`): Plain text files
- **Markdown** (`.md`): Markdown files
- **CSV** (`.csv`): Comma-separated values
- **Excel** (`.xlsx`, `.xls`): Excel spreadsheets

### File Type Specification

You can specify file types in several ways:

```bash
# By extension
--file-types .json,.txt,.csv

# By common names
--file-types json,text,excel

# Mixed
--file-types json,.md,csv
```

## Output

The script generates two types of output:

### 1. Summary CSV (default: `token_counts.csv`)
Contains one row per directory with:
- `directory`: Directory path
- `status`: Processing status (success/error)
- `total_tokens`: Total tokens in all files
- `file_count`: Number of files processed
- `error`: Error message (if any)

### 2. Detailed CSV (with `--detailed` flag)
Contains one row per file with:
- `directory`: Parent directory
- `file_path`: Relative file path
- `full_path`: Absolute file path
- `tokens`: Token count for this file
- `size_bytes`: File size in bytes
- `status`: Processing status

## Examples

### Example 1: Process current directory for JSONL files only
```bash
echo "." > my_paths.txt
uv run python token_counter.py --paths-file my_paths.txt --file-types jsonl
```

### Example 2: Process multiple directories with detailed output
```bash
# Create paths file
cat > data_paths.txt << EOF
/home/user/documents
/home/user/datasets
./local_data
EOF

# Run with detailed breakdown
uv run python token_counter.py --paths-file data_paths.txt --detailed --output detailed_results.csv
```

### Example 3: Process only text-based files with HuggingFace tokenizer
```bash
uv run python token_counter.py --tokenizer huggingface --hf-model bert-base-uncased --file-types txt,md,json --output text_tokens.csv
```

### Example 4: Compare tokenization between OpenAI and HuggingFace
```bash
# Count with tiktoken (GPT-4)
uv run python token_counter.py --tokenizer tiktoken --model gpt-4 --output tiktoken_results.csv

# Count with HuggingFace (BERT)
uv run python token_counter.py --tokenizer huggingface --hf-model bert-base-uncased --output hf_bert_results.csv

# Count with HuggingFace (GPT-2) 
uv run python token_counter.py --tokenizer huggingface --hf-model gpt2 --output hf_gpt2_results.csv
```

## Error Handling

The script handles various error conditions gracefully:

- **Missing directories**: Reports error but continues with other directories
- **Unsupported file types**: Skips with warning
- **Corrupted files**: Reports error for individual files but continues
- **Permission issues**: Reports access errors
- **Empty files**: Counts as 0 tokens

## Testing

Test the script with your existing JSONL file:

```bash
# This will process the combined_dataset.jsonl file in the current directory
uv run python token_counter.py
```

## Performance Notes

- Large files are processed in streaming fashion where possible
- Excel files (.xlsx) are loaded entirely into memory
- Progress is shown for directory processing
- Token counting is performed using OpenAI's optimized tiktoken library

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure UV dependencies are installed with `uv sync`
2. **File not found**: Check that paths in your paths file are correct
3. **Permission denied**: Ensure read access to directories and files
4. **Memory issues**: For very large Excel files, consider converting to CSV first

### Debug Mode

For troubleshooting, you can modify the script to add verbose logging by uncommenting debug lines or adding print statements. 
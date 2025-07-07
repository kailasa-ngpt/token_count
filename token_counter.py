#!/usr/bin/env python3
"""
Token Counter Script using OpenAI Tiktoken and HuggingFace Transformers

This script counts tokens in various file formats across multiple directories
and outputs the results to a CSV file.
"""

import os
import json
import csv
import pandas as pd
import tiktoken
import click
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TokenCounter:
    """Token counter class using OpenAI's tiktoken library or HuggingFace transformers."""
    
    def __init__(self, model_name: str = "gpt-4", tokenizer_type: str = "tiktoken", hf_model: Optional[str] = None, verbose: bool = False):
        """Initialize with either tiktoken or HuggingFace tokenizer.
        
        Args:
            model_name: Model name for tiktoken (e.g., 'gpt-4', 'gpt-3.5-turbo')
            tokenizer_type: Either 'tiktoken' or 'huggingface'
            hf_model: HuggingFace model name (e.g., 'bert-base-uncased', 'gpt2')
            verbose: Enable verbose logging for detailed progress
        """
        self.tokenizer_type = tokenizer_type
        self.verbose = verbose
        
        if tokenizer_type == "tiktoken":
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base encoding (used by gpt-4 and gpt-3.5-turbo)
                self.encoding = tiktoken.get_encoding("cl100k_base")
            self.tokenizer = None
            
        elif tokenizer_type == "huggingface":
            if not HF_AVAILABLE:
                raise ImportError("transformers library is required for HuggingFace tokenizers. Install with: pip install transformers")
            
            if not hf_model:
                raise ValueError("hf_model must be specified when using 'huggingface' tokenizer_type")
            
            # Suppress HuggingFace warnings that are not relevant for token counting
            import warnings
            warnings.filterwarnings("ignore", message=".*sequence length is longer than.*")
            warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
            warnings.filterwarnings("ignore", message=".*maximum sequence length.*")
            
            # Use AutoTokenizer with use_fast=True as requested
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
            self.encoding = None
            
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}. Use 'tiktoken' or 'huggingface'")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in given text."""
        if not text:
            return 0
            
        if self.tokenizer_type == "tiktoken":
            return len(self.encoding.encode(text))
        elif self.tokenizer_type == "huggingface":
            # Use encode method which returns token IDs, then count them
            if self.tokenizer is not None:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                return len(token_ids)
            else:
                raise ValueError("HuggingFace tokenizer is not initialized")
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")
    
    def process_json_file(self, file_path: Path) -> Tuple[int, str]:
        """Process JSON file and return token count."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
                return self.count_tokens(text), "success"
        except Exception as e:
            return 0, f"error: {str(e)}"
    
    def process_jsonl_file(self, file_path: Path) -> Tuple[int, str]:
        """Process JSONL file and return total token count."""
        try:
            total_tokens = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            text = json.dumps(data, ensure_ascii=False)
                            total_tokens += self.count_tokens(text)
                        except json.JSONDecodeError as e:
                            return 0, f"error at line {line_num}: {str(e)}"
            return total_tokens, "success"
        except Exception as e:
            return 0, f"error: {str(e)}"
    
    def process_text_file(self, file_path: Path) -> Tuple[int, str]:
        """Process text/markdown file and return token count."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                return self.count_tokens(text), "success"
        except Exception as e:
            return 0, f"error: {str(e)}"
    
    def process_csv_file(self, file_path: Path) -> Tuple[int, str]:
        """Process CSV file and return token count."""
        try:
            df = pd.read_csv(file_path)
            # Convert entire dataframe to string representation
            text = df.to_string(index=False)
            return self.count_tokens(text), "success"
        except Exception as e:
            return 0, f"error: {str(e)}"
    
    def process_xlsx_file(self, file_path: Path) -> Tuple[int, str]:
        """Process XLSX file and return token count."""
        try:
            # Read all sheets and combine
            xl_file = pd.ExcelFile(file_path)
            all_text = ""
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                all_text += df.to_string(index=False) + "\n"
            return self.count_tokens(all_text), "success"
        except Exception as e:
            return 0, f"error: {str(e)}"
    
    def process_file(self, file_path: Path) -> Tuple[int, str]:
        """Process a single file based on its extension."""
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return self.process_json_file(file_path)
        elif extension == '.jsonl':
            return self.process_jsonl_file(file_path)
        elif extension in ['.txt', '.md']:
            return self.process_text_file(file_path)
        elif extension == '.csv':
            return self.process_csv_file(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self.process_xlsx_file(file_path)
        else:
            return 0, f"unsupported file type: {extension}"


def read_directory_paths(paths_file: str) -> List[str]:
    """Read directory paths from a text file."""
    paths = []
    try:
        with open(paths_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    paths.append(line)
    except FileNotFoundError:
        click.echo(f"Error: Paths file '{paths_file}' not found.")
        return []
    except Exception as e:
        click.echo(f"Error reading paths file: {str(e)}")
        return []
    
    return paths


def get_supported_extensions(file_types: str) -> Set[str]:
    """Convert file types string to set of extensions."""
    if not file_types:
        return {'.json', '.jsonl', '.txt', '.md', '.csv', '.xlsx', '.xls'}
    
    # Map common file type names to extensions
    type_mapping = {
        'json': '.json',
        'jsonl': '.jsonl',
        'txt': '.txt',
        'text': '.txt',
        'md': '.md',
        'markdown': '.md',
        'csv': '.csv',
        'xlsx': '.xlsx',
        'xls': '.xls',
        'excel': ['.xlsx', '.xls']
    }
    
    extensions = set()
    types = [t.strip().lower() for t in file_types.split(',')]
    
    for file_type in types:
        if file_type in type_mapping:
            mapping = type_mapping[file_type]
            if isinstance(mapping, list):
                extensions.update(mapping)
            else:
                extensions.add(mapping)
        elif file_type.startswith('.'):
            extensions.add(file_type.lower())
    
    return extensions


def process_directory(directory: str, counter: TokenCounter, supported_extensions: Set[str], use_progress_bar: bool = True) -> Dict:
    """Process all supported files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return {
            'directory': directory,
            'status': 'error',
            'error': 'Directory does not exist',
            'total_tokens': 0,
            'file_count': 0,
            'files_processed': []
        }
    
    if not dir_path.is_dir():
        return {
            'directory': directory,
            'status': 'error',
            'error': 'Path is not a directory',
            'total_tokens': 0,
            'file_count': 0,
            'files_processed': []
        }
    
    total_tokens = 0
    files_processed = []
    
    # First collect all files to process
    files_to_process = []
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files_to_process.append(file_path)
    
    file_count = len(files_to_process)
    
    if file_count == 0:
        return {
            'directory': directory,
            'status': 'success',
            'error': '',
            'total_tokens': 0,
            'file_count': 0,
            'files_processed': []
        }
    
    # Process files with progress bar
    if use_progress_bar:
        progress_bar = tqdm(
            files_to_process, 
            desc=f"📂 {Path(directory).name}", 
            unit="file",
            leave=False,
            ncols=80
        )
    else:
        progress_bar = files_to_process
    
    for file_path in progress_bar:
        tokens, status = counter.process_file(file_path)
        total_tokens += tokens
        
        # Update progress bar description with current file and token info
        if use_progress_bar and isinstance(progress_bar, tqdm):
            file_name = file_path.name
            if len(file_name) > 20:
                file_name = file_name[:17] + "..."
            progress_bar.set_postfix(
                file=file_name,
                tokens=f"{tokens:,}" if status == "success" else "error"
            )
        
        files_processed.append({
            'file': str(file_path.relative_to(dir_path)),
            'full_path': str(file_path),
            'tokens': tokens,
            'status': status,
            'size_bytes': file_path.stat().st_size
        })
    
    if use_progress_bar and isinstance(progress_bar, tqdm):
        progress_bar.close()
    
    return {
        'directory': directory,
        'status': 'success',
        'error': '',
        'total_tokens': total_tokens,
        'file_count': file_count,
        'files_processed': files_processed
    }


@click.command()
@click.option('--paths-file', '-p', default='directory_paths.txt',
              help='Text file containing directory paths (one per line)')
@click.option('--file-types', '-t', default='',
              help='Comma-separated list of file types to process (e.g., json,txt,csv)')
@click.option('--output', '-o', default='token_counts.csv',
              help='Output CSV file name')
@click.option('--tokenizer', '--tokenizer-type', default='tiktoken',
              type=click.Choice(['tiktoken', 'huggingface'], case_sensitive=False),
              help='Tokenizer type to use: tiktoken or huggingface (default: tiktoken)')
@click.option('--model', '-m', default='gpt-4',
              help='Model name for tiktoken (e.g., gpt-4, gpt-3.5-turbo) (default: gpt-4)')
@click.option('--hf-model', default='',
              help='HuggingFace model name when using --tokenizer huggingface (e.g., bert-base-uncased, gpt2)')
@click.option('--detailed', '-d', is_flag=True,
              help='Include detailed file-by-file breakdown in output')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging (show file-level and directory-level progress)')
def main(paths_file: str, file_types: str, output: str, tokenizer: str, model: str, hf_model: str, detailed: bool, verbose: bool):
    """
    Count tokens in files across multiple directories using OpenAI Tiktoken or HuggingFace Transformers.
    
    This script reads directory paths from a text file and counts tokens
    in supported file formats, outputting results to a CSV file.
    """
    click.echo(f"🔢 Token Counter v0.1.0")
    click.echo(f"📁 Reading paths from: {paths_file}")
    click.echo(f"🔧 Using tokenizer: {tokenizer}")
    
    if tokenizer.lower() == "huggingface":
        if not hf_model:
            click.echo("❌ Error: --hf-model is required when using --tokenizer huggingface")
            click.echo("   Example: --tokenizer huggingface --hf-model bert-base-uncased")
            return
        click.echo(f"🤖 Using HuggingFace model: {hf_model}")
    else:
        click.echo(f"🤖 Using tiktoken model: {model}")
    
    # Initialize token counter
    try:
        if tokenizer.lower() == "huggingface":
            counter = TokenCounter(model_name=model, tokenizer_type="huggingface", hf_model=hf_model, verbose=verbose)
            click.echo(f"✅ Initialized HuggingFace tokenizer for model: {hf_model} (use_fast=True)")
        else:
            counter = TokenCounter(model_name=model, tokenizer_type="tiktoken", verbose=verbose)
            click.echo(f"✅ Initialized tiktoken for model: {model}")
    except Exception as e:
        click.echo(f"❌ Error initializing tokenizer: {str(e)}")
        return
    
    # Read directory paths
    directories = read_directory_paths(paths_file)
    if not directories:
        click.echo("❌ No valid directories found in paths file.")
        return
    
    click.echo(f"📂 Found {len(directories)} directories to process")
    
    # Get supported file extensions
    supported_extensions = get_supported_extensions(file_types)
    if file_types:
        click.echo(f"📄 Processing file types: {', '.join(sorted(supported_extensions))}")
    else:
        click.echo(f"📄 Processing all supported file types: {', '.join(sorted(supported_extensions))}")
    
    # Process each directory
    results = []
    detailed_results = []
    
    # Process directories with progress bars
    # Use tqdm for directory-level progress bar
    with tqdm(directories, desc="🗂️  Directories", unit="dir", ncols=100) as dir_progress:
        for directory in dir_progress:
            dir_name = Path(directory).name
            if len(dir_name) > 30:
                dir_name = dir_name[:27] + "..."
            dir_progress.set_description(f"🗂️  Processing: {dir_name}")
            
            result = process_directory(directory, counter, supported_extensions, use_progress_bar=True)
            results.append(result)
            
            # Update directory progress with results
            if result['status'] == 'success':
                dir_progress.set_postfix(
                    files=result['file_count'],
                    tokens=f"{result['total_tokens']:,}"
                )
            else:
                dir_progress.set_postfix(error=result['error'][:30])
            
            if detailed and result['files_processed']:
                for file_info in result['files_processed']:
                    detailed_results.append({
                        'directory': directory,
                        'file_path': file_info['file'],
                        'full_path': file_info['full_path'],
                        'tokens': file_info['tokens'],
                        'size_bytes': file_info['size_bytes'],
                        'status': file_info['status']
                    })
    
    # Create summary DataFrame
    summary_data = []
    total_tokens_all = 0
    total_files_all = 0
    
    for result in results:
        summary_data.append({
            'directory': result['directory'],
            'status': result['status'],
            'total_tokens': result['total_tokens'],
            'file_count': result['file_count'],
            'error': result['error']
        })
        
        if result['status'] == 'success':
            total_tokens_all += result['total_tokens']
            total_files_all += result['file_count']
    
    # Save summary results
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output, index=False)
    
    # Save detailed results if requested
    if detailed and detailed_results:
        detailed_output = output.replace('.csv', '_detailed.csv')
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(detailed_output, index=False)
        click.echo(f"📊 Detailed results saved to: {detailed_output}")
    
    # Print summary
    click.echo(f"\n📈 Summary:")
    click.echo(f"   Total directories processed: {len(directories)}")
    click.echo(f"   Total files processed: {total_files_all}")
    click.echo(f"   Total tokens counted: {total_tokens_all:,}")
    click.echo(f"   Results saved to: {output}")
    
    # Show top directories by token count
    if len(results) > 1:
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            top_dirs = sorted(successful_results, key=lambda x: x['total_tokens'], reverse=True)[:5]
            click.echo(f"\n🏆 Top directories by token count:")
            for i, result in enumerate(top_dirs, 1):
                click.echo(f"   {i}. {result['directory']}: {result['total_tokens']:,} tokens ({result['file_count']} files)")


if __name__ == '__main__':
    main() 
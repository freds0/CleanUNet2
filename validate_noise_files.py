#!/usr/bin/env python3
"""
Script to validate audio files in a noise dataset and create a list of valid files.
This helps prevent training crashes due to corrupted or incompatible audio files.

Usage:
    python validate_noise_files.py --input_dir /path/to/noises --output valid_noise_files.txt
"""

import argparse
import torchaudio
from pathlib import Path
from tqdm import tqdm


def validate_audio_file(file_path, min_duration_sec=0.1):
    """
    Valida se um arquivo de áudio pode ser carregado e tem duração mínima.

    Args:
        file_path (str): Caminho do arquivo de áudio
        min_duration_sec (float): Duração mínima em segundos

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        info = torchaudio.info(str(file_path))
        duration = info.num_frames / info.sample_rate

        if duration < min_duration_sec:
            return False, f"Too short ({duration:.3f}s < {min_duration_sec}s)"

        # Check for reasonable sample rate
        if info.sample_rate < 8000 or info.sample_rate > 192000:
            return False, f"Invalid sample rate ({info.sample_rate} Hz)"

        # Check for valid number of channels
        if info.num_channels < 1:
            return False, "No audio channels"

        return True, None

    except Exception as e:
        return False, str(e)


def get_audio_files_recursively(directory, extensions=('.wav', '.flac', '.mp3', '.ogg')):
    """
    Busca recursivamente por arquivos de áudio em um diretório.

    Args:
        directory (str): Caminho do diretório para buscar
        extensions (tuple): Extensões de arquivos de áudio a serem buscados

    Returns:
        list: Lista de caminhos absolutos dos arquivos encontrados
    """
    audio_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return audio_files

    print(f"Searching for audio files in: {directory}")
    for ext in extensions:
        pattern = f"**/*{ext}"
        audio_files.extend([str(f) for f in directory_path.glob(pattern)])

    print(f"Found {len(audio_files)} audio files")
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio files and create a list of valid files"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing audio files to validate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="valid_noise_files.txt",
        help="Output file to save list of valid files",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--log_errors",
        type=str,
        default=None,
        help="Optional file to log invalid files and their errors",
    )
    args = parser.parse_args()

    # Find all audio files
    audio_files = get_audio_files_recursively(args.input_dir)

    if not audio_files:
        print("No audio files found!")
        return

    # Validate files
    print(f"\nValidating {len(audio_files)} audio files...")
    valid_files = []
    invalid_files = []

    for file_path in tqdm(audio_files, desc="Validating"):
        is_valid, error_msg = validate_audio_file(file_path, args.min_duration)

        if is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append((file_path, error_msg))

    # Print summary
    print(f"\n{'='*70}")
    print(f"Validation Complete")
    print(f"{'='*70}")
    print(f"Total files scanned:  {len(audio_files)}")
    print(f"Valid files:          {len(valid_files)} ({100*len(valid_files)/len(audio_files):.1f}%)")
    print(f"Invalid files:        {len(invalid_files)} ({100*len(invalid_files)/len(audio_files):.1f}%)")
    print(f"{'='*70}\n")

    # Save valid files list
    with open(args.output, "w") as f:
        for file_path in valid_files:
            f.write(f"{file_path}\n")
    print(f"Valid files list saved to: {args.output}")

    # Optionally save error log
    if args.log_errors and invalid_files:
        with open(args.log_errors, "w") as f:
            f.write("Invalid Audio Files Report\n")
            f.write("=" * 70 + "\n\n")
            for file_path, error_msg in invalid_files:
                f.write(f"File: {file_path}\n")
                f.write(f"Error: {error_msg}\n")
                f.write("-" * 70 + "\n")
        print(f"Error log saved to: {args.log_errors}")

    # Show some examples of invalid files
    if invalid_files:
        print(f"\nExample invalid files (showing first 10):")
        for file_path, error_msg in invalid_files[:10]:
            print(f"  - {Path(file_path).name}: {error_msg}")


if __name__ == "__main__":
    main()

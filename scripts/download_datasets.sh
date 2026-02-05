#!/bin/bash
# Script para baixar datasets comuns para treinamento com augmentation
# Uso: bash scripts/download_datasets.sh [dataset_name] [output_dir]

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para imprimir mensagens
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Função de ajuda
show_help() {
    cat << EOF
Uso: $0 [DATASET] [OUTPUT_DIR]

Datasets disponíveis:
  ljspeech       - LJSpeech-1.1 (~2.6 GB)
  demand         - DEMAND noise dataset (~1.5 GB)
  wham-noise     - WHAM! noise dataset (~6 GB)
  all-speech     - Baixa apenas datasets de fala (LJSpeech)
  all-noise      - Baixa apenas datasets de ruído (DEMAND + WHAM)
  all            - Baixa tudo

Exemplos:
  $0 ljspeech /datasets
  $0 demand /datasets/noise
  $0 all /datasets

EOF
    exit 0
}

# Verifica argumentos
if [ "$#" -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

DATASET=$1
OUTPUT_DIR=${2:-.}  # Default: diretório atual

# Cria diretório de saída
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

print_info "Diretório de saída: $(pwd)"

# Função para baixar LJSpeech
download_ljspeech() {
    print_info "Baixando LJSpeech-1.1 (~2.6 GB)..."

    if [ -d "LJSpeech-1.1" ]; then
        print_warning "LJSpeech-1.1 já existe. Pulando download."
        return 0
    fi

    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

    print_info "Extraindo LJSpeech-1.1..."
    tar -xvjf LJSpeech-1.1.tar.bz2

    print_info "Limpando arquivo temporário..."
    rm LJSpeech-1.1.tar.bz2

    print_info "✓ LJSpeech baixado e extraído em: $(pwd)/LJSpeech-1.1"
    print_info "  Arquivos de áudio: $(pwd)/LJSpeech-1.1/wavs/"
}

# Função para baixar DEMAND
download_demand() {
    print_info "Baixando DEMAND noise dataset (~1.5 GB)..."

    if [ -d "DEMAND" ]; then
        print_warning "DEMAND já existe. Pulando download."
        return 0
    fi

    wget https://zenodo.org/record/1227121/files/DEMAND.zip

    print_info "Extraindo DEMAND..."
    unzip -q DEMAND.zip

    print_info "Limpando arquivo temporário..."
    rm DEMAND.zip

    print_info "✓ DEMAND baixado e extraído em: $(pwd)/DEMAND"
}

# Função para baixar WHAM! noise
download_wham_noise() {
    print_info "Baixando WHAM! noise dataset (~6 GB)..."

    if [ -d "wham_noise" ]; then
        print_warning "wham_noise já existe. Pulando download."
        return 0
    fi

    wget https://storage.googleapis.com/whisper-public/wham_noise.zip

    print_info "Extraindo WHAM! noise..."
    unzip -q wham_noise.zip

    print_info "Limpando arquivo temporário..."
    rm wham_noise.zip

    print_info "✓ WHAM! noise baixado e extraído em: $(pwd)/wham_noise"
}

# Verifica qual dataset baixar
case "$DATASET" in
    ljspeech)
        download_ljspeech
        ;;
    demand)
        download_demand
        ;;
    wham-noise)
        download_wham_noise
        ;;
    all-speech)
        print_info "Baixando todos os datasets de fala..."
        download_ljspeech
        ;;
    all-noise)
        print_info "Baixando todos os datasets de ruído..."
        download_demand
        download_wham_noise
        ;;
    all)
        print_info "Baixando todos os datasets..."
        download_ljspeech
        download_demand
        download_wham_noise
        ;;
    *)
        print_error "Dataset desconhecido: $DATASET"
        echo ""
        show_help
        ;;
esac

# Resumo final
echo ""
echo "========================================================================"
print_info "Download concluído!"
echo "========================================================================"
echo ""
echo "Datasets baixados em: $(pwd)"
echo ""

if [ -d "LJSpeech-1.1" ]; then
    echo "✓ LJSpeech-1.1/wavs/ (~13100 arquivos de áudio limpo)"
fi

if [ -d "DEMAND" ]; then
    echo "✓ DEMAND/ (16 tipos de ruído ambiental)"
fi

if [ -d "wham_noise" ]; then
    echo "✓ wham_noise/ (ruídos de ambientes diversos)"
fi

echo ""
echo "Próximos passos:"
echo "1. Edite configs/train_ljspeech_augmentation.yaml"
echo "2. Ajuste os caminhos:"
if [ -d "LJSpeech-1.1" ]; then
    echo "   data_dir: \"$(pwd)/LJSpeech-1.1\""
fi
if [ -d "DEMAND" ]; then
    echo "   background_paths: \"$(pwd)/DEMAND\""
fi
if [ -d "wham_noise" ]; then
    echo "   # ou background_paths: \"$(pwd)/wham_noise\""
fi
echo "3. Execute: python train.py --config configs/train_ljspeech_augmentation.yaml"
echo ""

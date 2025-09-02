# AASIST for various ASVspoof datasets

This repository is a modified version of the original [AASIST (clovaai/aasist)](https://github.com/clovaai/aasist) project, adapted to work flexibly with various ASVspoof datasets (e.g., ASVspoof 2019, ASVspoof 2021, and ASVspoof 5).



## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/d0v0h/AASIST-for-various-ASVspoof-datasets.git
    cd AASIST-for-various-ASVspoof-datasets
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

1.  **Download the datasets:**
    Download the required ASVspoof dataset(s) from the links below.

    <details>
    <summary><b>🔗 Click to see download links</b></summary>

    | Dataset             | Link                                                                                             |
    |---------------------|--------------------------------------------------------------------------------------------------|
    | ASVspoof 2019 LA    | https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y             |
    | ASVspoof 2021 LA    | https://zenodo.org/records/4837263                                                               |
    | 21LA keys           | https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz                                         |
    | ASVspoof 2021 DF    | https://zenodo.org/records/4835108                                                               |
    | 21DF keys           | https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz                                         |
    | ASVspoof 5          | https://zenodo.org/records/14498691                                                              |

    </details>

2.  **Prepare the directory structure:**
    Place the downloaded datasets inside a `DB/` directory at the project root, following the structure below.

    <details>
    <summary><b>📁 Click to see the expected database structure</b></summary>

    ```
    DB
    ├── ASVspoof2019
    │   ├── ASVspoof2019_LA_asv_protocols
    │   ├── ASVspoof2019_LA_asv_scores
    │   ├── ASVspoof2019_LA_cm_protocols
    │   ├── ASVspoof2019_LA_dev
    │   │   └── flac
    │   ├── ASVspoof2019_LA_eval
    │   │   └── flac
    │   └── ASVspoof2019_LA_train
    │       └── flac
    ├── ASVspoof2021
    │   ├── ASVspoof2021_DF_eval
    │   │   └── flac
    │   ├── ASVspoof2021_LA_eval
    │   │   └── flac
    │   └── keys
    │       ├── DF
    │       │   └── CM
    │       └── LA
    │           ├── ASV
    │           └── CM
    └── ASVspoof5
        ├── flac_D
        ├── flac_E_eval
        └── flac_T
    ```
    </details>

## Usage

### Training

To train AASIST on different datasets, use the corresponding script:

*   **ASVspoof 2019 LA:**
    ```bash
    python main_2019.py --config ./config/AASIST.conf
    ```

*   **ASVspoof 2021 LA:**
    ```bash
    python main_2021.py --config ./config/AASIST.conf --track LA
    ```

*   **ASVspoof 2021 DF:**
    ```bash
    python main_2021.py --config ./config/AASIST.conf --track DF
    ```

*   **ASVspoof 5:**
    ```bash
    python main_asv5.py --config ./config/AASIST.conf
    ```

### Evaluation

To evaluate a trained model, use the `--eval` flag. For example:

```bash
python main_2021.py --config ./config/AASIST.conf --track LA --eval
```

## License

This project is based on the original AASIST, which is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

**Important**: This repository contains subcomponents with different licenses. Please review the [NOTICE](NOTICE) file for details on all included licenses, as some components have restrictions
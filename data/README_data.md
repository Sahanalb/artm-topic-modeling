# Dataset

## Amazon Musical Instruments Reviews

This project uses the **Amazon Product Reviews — Musical Instruments (5-core)** dataset
introduced by He & McAuley (2016).

### Download

1. Go to: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
2. Find **"Musical Instruments"** under the 5-core datasets
3. Download `Musical_Instruments_5.json`
4. Place it in this `data/` folder

### Convert to JSONL

```bash
python src/convert_to_jsonl.py
```

This produces `data/amazon_musical_instruments_reviews.jsonl` with 10,206 reviews
in the format:
```json
{"id": 1, "asin": "...", "rating": 5.0, "summary": "...", "review": "..."}
```

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total reviews (raw) | 10,254 |
| Reviews after filtering (len > 40 chars) | 10,206 |
| Average review length | 187 words |
| Rating distribution | 1★–5★ |
| Product categories | Guitars, Amps, Cables, Tuners, Mics, Stands, Accessories |

### Citation

```bibtex
@inproceedings{he2016ups,
  title     = {Ups and Downs: Modeling the Visual Evolution of Fashion Trends
               with One-Class Collaborative Filtering},
  author    = {He, Ruining and McAuley, Julian},
  booktitle = {Proceedings of the 25th International Conference on World Wide Web},
  pages     = {507--517},
  year      = {2016}
}
```

### Note on Data Privacy

This dataset contains publicly posted Amazon reviews. No personally identifiable
information beyond Amazon userIDs is included. The dataset is used strictly for
academic research purposes.

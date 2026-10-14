old_emb=$(find data/processed/vlm_emb -name '*.npz' | wc -l)
new_emb=$(find data/processed/vlm_emb_violence -name '*.npz' 2>/dev/null | wc -l)

echo "Embeddings Progress: $new_emb / $old_emb ($(awk "BEGIN {printf \"%.1f\", 100*$new_emb/$old_emb}")%)"

old_labels=$(find data/processed/vlm_labels -name '*.jsonl' | wc -l)
new_labels=$(find data/processed/vlm_labels_violence -name '*.jsonl' 2>/dev/null | wc -l)

echo "Labels Progress: $new_labels / $old_labels ($(awk "BEGIN {printf \"%.1f\", 100*$new_labels/$old_labels}")%)"
from src.modeling import train_and_save

if __name__ == "__main__":
    name, comparison, test_metrics = train_and_save()
    print("\nValidation model comparison (sorted by MAE):")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {name}")
    print("Held-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")
    print("\nSaved model and reports in ./artifacts")

"""
Interactive Demo: Ethanol-Water Distillation Column Surrogate Model

Instead of opening DWSIM, building the flowsheet, and waiting for it to solve,
type in a feed composition and reflux ratio and get instant predictions.

Usage:
    python demo.py
    (then follow the prompts)

    OR run non-interactively:
    python demo.py --feed 0.4 --reflux 5
"""
import argparse
import os
import joblib
import pandas as pd
import sys

VALID_FEED_RANGE = (0.3, 0.5)
VALID_REFLUX_RANGE = (2, 8)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_models.pkl')

def load_models(path=DEFAULT_MODEL_PATH):
    return joblib.load(path)

def predict(models, feed_x, reflux):
    X = pd.DataFrame({'feed_ethanol_x': [feed_x], 'reflux_ratio': [reflux]})
    out = {}
    for target, model in models.items():
        out[target] = model.predict(X)[0]
    return out

def warn_if_out_of_range(feed_x, reflux):
    warnings = []
    if not (VALID_FEED_RANGE[0] <= feed_x <= VALID_FEED_RANGE[1]):
        warnings.append(f"  Feed composition {feed_x} is outside the validated range "
                         f"{VALID_FEED_RANGE} — prediction may be unreliable (extrapolation).")
    if not (VALID_REFLUX_RANGE[0] <= reflux <= VALID_REFLUX_RANGE[1]):
        warnings.append(f"  Reflux ratio {reflux} is outside the validated range "
                         f"{VALID_REFLUX_RANGE} — prediction may be unreliable (extrapolation).")
    return warnings

def print_result(feed_x, reflux, result, warnings):
    print("\n" + "="*55)
    print(f"  Feed ethanol mole fraction : {feed_x}")
    print(f"  Reflux ratio               : {reflux}")
    print("-"*55)
    print(f"  Distillate ethanol x       : {result['distillate_ethanol_x']:.4f}")
    print(f"  Bottoms water mass frac    : {result['bottoms_water_massfrac']:.4f}")
    print(f"  Condenser duty             : {result['condenser_duty_kW']:.1f} kW")
    print(f"  Reboiler duty              : {result['reboiler_duty_kW']:.1f} kW")
    print("="*55)
    for w in warnings:
        print(f"  \u26a0  {w}")
    if warnings:
        print()

def main():
    parser = argparse.ArgumentParser(description="Ethanol-water distillation surrogate model demo")
    parser.add_argument('--feed', type=float, help='Feed ethanol mole fraction (e.g. 0.4)')
    parser.add_argument('--reflux', type=float, help='Reflux ratio (e.g. 5)')
    args = parser.parse_args()

    models = load_models()

    if args.feed is not None and args.reflux is not None:
        warnings = warn_if_out_of_range(args.feed, args.reflux)
        result = predict(models, args.feed, args.reflux)
        print_result(args.feed, args.reflux, result, warnings)
        return

    print("Ethanol-Water Distillation Column Surrogate Model")
    print(f"Validated range: feed x in {VALID_FEED_RANGE}, reflux ratio in {VALID_REFLUX_RANGE}")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        feed_in = input("Feed ethanol mole fraction: ").strip()
        if feed_in.lower() == 'q':
            break
        reflux_in = input("Reflux ratio: ").strip()
        if reflux_in.lower() == 'q':
            break
        try:
            feed_x = float(feed_in)
            reflux = float(reflux_in)
        except ValueError:
            print("Please enter numeric values.\n")
            continue

        warnings = warn_if_out_of_range(feed_x, reflux)
        result = predict(models, feed_x, reflux)
        print_result(feed_x, reflux, result, warnings)

if __name__ == '__main__':
    main()
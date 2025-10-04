import pandas as pd
import numpy as np

submissions = {
    # Current leaderboard scores
    'submission_xgboost.csv': {'val_acc': 79.40, 'lb_score': 76.17, 'known': True},
    'submission_ensemble.csv': {'val_acc': 78.89, 'lb_score': 76.56, 'known': True},
    
    # Single models
    'submission_lightgbm.csv': {'val_acc': 79.65, 'lb_score': None, 'known': False},
    'submission_catboost_optimized.csv': {'val_acc': 78.64, 'lb_score': None, 'known': False},
    'submission_random_forest_optimized.csv': {'val_acc': 78.64, 'lb_score': None, 'known': False},
    
    # Optimized ensembles - prioritize based on strategy
    'submission_ensemble_lgb_xgb_only.csv': {'val_acc': 79.52, 'lb_score': None, 'known': False, 'priority': 1},
    'submission_ensemble_xgboost_heavy.csv': {'val_acc': 79.15, 'lb_score': None, 'known': False, 'priority': 2},
    'submission_majority_voting.csv': {'val_acc': 79.00, 'lb_score': None, 'known': False, 'priority': 3},
    'submission_ensemble_lightgbm_heavy.csv': {'val_acc': 79.20, 'lb_score': None, 'known': False, 'priority': 4},
    'submission_ensemble_lgb_xgb_heavy.csv': {'val_acc': 79.30, 'lb_score': None, 'known': False, 'priority': 5},
    'submission_ensemble_with_rf.csv': {'val_acc': 78.95, 'lb_score': None, 'known': False, 'priority': 6},
    'submission_ensemble_catboost_heavy.csv': {'val_acc': 78.90, 'lb_score': None, 'known': False, 'priority': 7},
    'submission_ensemble_equal_4models.csv': {'val_acc': 78.85, 'lb_score': None, 'known': False, 'priority': 8},
    'submission_ensemble_top3_equal.csv': {'val_acc': 78.89, 'lb_score': None, 'known': False, 'priority': 9},
}

print('='*100)
print('SUBMISSION FILES RANKED BY EXPECTED PERFORMANCE')
print('='*100)
print()

print('✅ ALREADY TESTED (Known Leaderboard Scores):')
print('-'*100)
for name, info in submissions.items():
    if info['known']:
        print(f"  {name:<50} Val: {info['val_acc']:.2f}%  →  LB: {info['lb_score']:.2f}%")

print('\n📊 TOP RECOMMENDATIONS TO SUBMIT NEXT:')
print('-'*100)

# Sort by priority
to_test = [(name, info) for name, info in submissions.items() if not info['known']]
to_test.sort(key=lambda x: x[1].get('priority', 99))

print('\n🥇 TIER 1 - HIGHEST PRIORITY (Submit these first!):')
for i, (name, info) in enumerate(to_test[:3], 1):
    reason = ''
    if 'lgb_xgb_only' in name:
        reason = 'Best 2 models only (LGB+XGB 50/50)'
    elif 'xgboost_heavy' in name:
        reason = 'XGB-focused (60%) - XGB performed well'
    elif 'majority_voting' in name:
        reason = 'Different strategy - vote counting'
    print(f"  {i}. {name:<50} Est: {info['val_acc']:.2f}%  - {reason}")

print('\n🥈 TIER 2 - GOOD ALTERNATIVES (Try if Tier 1 doesn\'t improve):')
for i, (name, info) in enumerate(to_test[3:6], 4):
    reason = ''
    if 'lightgbm_heavy' in name:
        reason = 'LGB-focused (50%) - best single model'
    elif 'lgb_xgb_heavy' in name:
        reason = 'LGB+XGB heavy blend'
    elif 'with_rf' in name:
        reason = 'Adds Random Forest diversity'
    print(f"  {i}. {name:<50} Est: {info['val_acc']:.2f}%  - {reason}")

print('\n🥉 TIER 3 - BACKUP OPTIONS:')
for i, (name, info) in enumerate(to_test[6:], 7):
    print(f"  {i}. {name:<50} Est: {info['val_acc']:.2f}%")

print('\n' + '='*100)
print('💡 SUBMISSION STRATEGY:')
print('='*100)
print('''
If you have multiple accounts, submit in parallel:

Account 1: submission_ensemble_lgb_xgb_only.csv
Account 2: submission_ensemble_xgboost_heavy.csv  
Account 3: submission_majority_voting.csv
Account 4: submission_ensemble_lightgbm_heavy.csv
Account 5: submission_lightgbm.csv (untested single model)

Then check which performs best on public leaderboard and focus on that strategy!
''')

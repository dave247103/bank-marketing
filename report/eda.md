# EDA Summary

Dataset size:
- Rows: 45,211
- Columns: 17 (16 features + label `y`)

Schema summary:
- Numeric: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- Categorical: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- Label: `y` (binary)

Label distribution:
- `no`: 39,922 (88.30%)
- `yes`: 5,289 (11.70%)

Notable categorical cardinalities (unique values):
- `job`: 12
- `month`: 12
- `education`: 4
- `poutcome`: 4
- `marital`: 3
- `contact`: 3

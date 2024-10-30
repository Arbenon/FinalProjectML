import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
# Завантаження даних
train_file_path = '/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv'
test_file_path = '/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv'
sample_submission_path = '/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_sample_submission.csv'


train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(sample_submission_path)

# Попередня перевірка наборів даних
print("Train Data Sample:")
print(train_data.head())
print("\nTest Data Sample:")
print(test_data.head())
print("\nSample Submission:")
print(sample_submission.head())


# %%
# One-Hot Encoding для ознак з малою кількістю унікальних значень
one_hot_columns = ['Var194', 'Var196', 'Var203', 'Var205', 'Var208', 'Var210', 'Var211', 'Var218', 'Var221', 'Var223', 'Var225', 'Var227']
train_data = pd.get_dummies(train_data, columns=one_hot_columns, drop_first=True)
test_data = pd.get_dummies(test_data, columns=one_hot_columns, drop_first=True)

# Синхронізація колонок тестового набору з тренувальним
train_columns = train_data.columns
for col in train_columns:
    if col not in test_data.columns:
        test_data[col] = 0
test_data = test_data[train_columns]
print('Ready!')



# %%
# Видалення колонок з більш ніж 90% пропусків
threshold = 0.90
train_data = train_data.loc[:, train_data.isnull().mean() < threshold]
test_data = test_data.loc[:, test_data.isnull().mean() < threshold]

# Заповнення пропусків тільки у числових колонках медіаною
numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
train_data[numeric_columns] = imputer.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = imputer.transform(test_data[numeric_columns])
print('Ready!')



# %%
# Відокремлюємо цільову змінну та ознаки
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
X_test = test_data.drop(columns=['y'])

# Переконаємося, що всі дані числові
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Синхронізація колонок після кодування
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print('Ready!')





# %%
# Розбиваємо тренувальний набір на тренувальні і валідаційні дані
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Перетворення цільової змінної на цілі значення
y_train_split = y_train_split.round().astype(int)
y_valid = y_valid.round().astype(int)
print('Ready!')






# %%
# AdaBoost Classifier
ada_clf = AdaBoostClassifier(n_estimators=110, learning_rate=1.0, random_state=42)
ada_clf.fit(X_train_split, y_train_split)
y_pred_ada = ada_clf.predict(X_valid)
ada_accuracy = balanced_accuracy_score(y_valid, y_pred_ada)
print("\nТочність AdaBoost на валідаційному наборі (Balanced Accuracy):", ada_accuracy)




# Прогнозування на тестовому наборі
y_test_pred_ada = ada_clf.predict(X_test)

# Підготовка файлу submission на основі шаблону
submission = sample_submission.copy()
submission['y'] = y_test_pred_ada  # Замініть 'y' на назву стовпця для прогнозу в sample_submission, якщо вона інша

# Збереження файлу для завантаження на Kaggle
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv створено і готовий для сабміту!")
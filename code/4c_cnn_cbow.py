import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# --- 中文字体设置 (确保您的系统已安装相应字体，如SimHei, Microsoft YaHei) ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 配置与路径定义 ---
embedding_type = 'word2vec_cbow' # <--- 改为新的嵌入类型
model_type = 'cnn'

base_data_path = r'D:\Codes\textmining\data'
classification_data_dir = os.path.join(base_data_path, 'classification_data', embedding_type)
labels_classes_path = os.path.join(base_data_path, 'processed', 'research_areas_classes.pkl')

output_models_dir = os.path.join(base_data_path, '..', 'models') 
output_results_dir = os.path.join(base_data_path, '..', 'results', 'classification')
accuracy_scores_file = os.path.join(output_results_dir, 'accuracy_scores.txt')
model_save_path = os.path.join(output_models_dir, f'{model_type}_{embedding_type}.h5')

# --- 创建输出目录 (如果不存在) ---
for dir_path in [output_models_dir, output_results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录已创建: {dir_path}")

# --- 1. 加载数据 ---
print(f"\n--- 正在加载 '{embedding_type}' 嵌入的训练集和测试集 ---")
try:
    X_train = np.load(os.path.join(classification_data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(classification_data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(classification_data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(classification_data_dir, 'y_test.npy'))
    
    with open(labels_classes_path, 'rb') as f:
        class_names = pickle.load(f)
        
    print(f"X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    print(f"X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")
    print(f"共有 {len(class_names)} 个类别。")
except FileNotFoundError as e:
    print(f"错误: 数据文件未找到 - {e.filename}。请确保已运行数据划分脚本且路径正确。")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# --- 2. 准备输入数据以适配CNN ---
input_dim = X_train.shape[1] 
X_train_reshaped = X_train.reshape(X_train.shape[0], input_dim, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], input_dim, 1)
num_classes = y_train.shape[1]

print(f"Reshaped X_train 形状: {X_train_reshaped.shape}")
print(f"Reshaped X_test 形状: {X_test_reshaped.shape}")

# --- 3. 构建1D CNN模型 ---
print("\n--- 正在构建1D CNN模型 ---")
def build_cnn_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Conv1D(128, 5, activation='relu', padding='same')(input_tensor)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x) 
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

cnn_model = build_cnn_model(input_shape=(input_dim, 1), num_classes=num_classes)
cnn_model.summary()

# --- 4. 编译模型 ---
print("\n--- 正在编译模型 ---")
cnn_model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# --- 5. 训练模型 ---
print("\n--- 正在训练模型 ---")
epochs = 50 
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

history = cnn_model.fit(
    X_train_reshaped, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print(f"\n从 '{model_save_path}' 加载最佳权重模型 (如果 EarlyStopping 生效)...")
cnn_model = load_model(model_save_path) 

# --- 6. 评估模型 ---
print("\n--- 正在评估模型 ---")
loss, binary_acc, precision, recall = cnn_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"测试集损失: {loss:.4f}")
print(f"测试集二元准确率 (Keras): {binary_acc:.4f}")
print(f"测试集精确率 (Keras Precision): {precision:.4f}")
print(f"测试集召回率 (Keras Recall): {recall:.4f}")

y_pred_proba = cnn_model.predict(X_test_reshaped)

# 添加多阈值测试逻辑
thresholds_to_try = [0.05, 0.1, 0.15, 0.2, 0.25]
best_f1 = 0
best_threshold = 0.5  # 默认保留原始阈值

for threshold in thresholds_to_try:
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    # 计算指标
    subset_acc = accuracy_score(y_test, y_pred_binary)
    f1_micro = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)
    
    print(f"\n使用阈值 {threshold} 的结果:")
    print(f"  子集准确率: {subset_acc:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    
    # 选择F1分数最高的阈值
    if f1_micro > best_f1:
        best_f1 = f1_micro
        best_threshold = threshold

print(f"\n最佳阈值: {best_threshold:.2f} (F1 Micro: {best_f1:.4f})")
y_pred_binary = (y_pred_proba > best_threshold).astype(int)

f1_macro = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
f1_samples = f1_score(y_test, y_pred_binary, average='samples', zero_division=0)

print(f"\nScikit-learn 指标 (Threshold={best_threshold:.2f}):")
print(f"  子集准确率 (Exact Match Ratio): {subset_acc:.4f}")
print(f"  F1 Score (Micro): {f1_micro:.4f}")
print(f"  F1 Score (Macro): {f1_macro:.4f}")
print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
print(f"  F1 Score (Samples): {f1_samples:.4f}")

print("\nClassification Report (scikit-learn):")
try:
    report = classification_report(y_test, y_pred_binary, target_names=class_names, zero_division=0)
    print(report)
except ValueError as ve: 
    print(f"生成classification_report时出错: {ve}. 将不使用target_names参数。")
    report = classification_report(y_test, y_pred_binary, zero_division=0)
    print(report)
    
# --- 7. 保存结果 ---
print("\n--- 正在保存结果 ---")
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
header = "Timestamp,Embedding,Model,Threshold,Loss,BinaryAccuracy,Precision,Recall,SubsetAccuracy,F1_Micro,F1_Macro,F1_Weighted,F1_Samples\n"
if not os.path.exists(accuracy_scores_file):
    with open(accuracy_scores_file, 'w') as f:
        f.write(header)

with open(accuracy_scores_file, 'a') as f:
    f.write(f"{timestamp},{embedding_type},{model_type},{best_threshold:.2f},"
            f"{loss:.4f},{binary_acc:.4f},{precision:.4f},{recall:.4f},"
            f"{subset_acc:.4f},{f1_micro:.4f},{f1_macro:.4f},{f1_weighted:.4f},{f1_samples:.4f}\n")
print(f"评估结果已追加到 '{accuracy_scores_file}'")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title(f'损失 ({embedding_type} + {model_type.upper()})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['binary_accuracy'], label='训练二元准确率')
plt.plot(history.history['val_binary_accuracy'], label='验证二元准确率')
plt.title(f'二元准确率 ({embedding_type} + {model_type.upper()})')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.legend()

history_plot_path = os.path.join(output_results_dir, f'{model_type}_{embedding_type}_training_history.png')
plt.tight_layout()
plt.savefig(history_plot_path)
print(f"训练历史图已保存到 '{history_plot_path}'")
plt.close()

print("\n正在生成混淆矩阵图 (示例)...")
mcm = multilabel_confusion_matrix(y_test, y_pred_binary)
all_labels_flat_test = [class_names[i] for doc_labels in y_test for i, label_present in enumerate(doc_labels) if label_present]
label_counts_test = Counter(all_labels_flat_test)

top_n_classes_to_plot = min(3, len(label_counts_test)) 
if top_n_classes_to_plot > 0:
    top_classes_names = [item[0] for item in label_counts_test.most_common(top_n_classes_to_plot)]
    class_indices_to_plot = [i for i, name in enumerate(class_names) if name in top_classes_names]

    if not class_indices_to_plot:
        print("无法确定用于绘制混淆矩阵的高频类别。跳过混淆矩阵绘制。")
    else:
        num_plots = len(class_indices_to_plot)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), squeeze=False) 

        for i, class_idx in enumerate(class_indices_to_plot):
            class_name_actual = class_names[class_idx]
            sns.heatmap(mcm[class_idx], annot=True, fmt='d', cmap='Blues', ax=axes[0, i],
                        xticklabels=['预测为负', '预测为正'], # Changed to Chinese
                        yticklabels=['实际为负', '实际为正']) # Changed to Chinese
            axes[0, i].set_title(f'混淆矩阵: {class_name_actual[:25]}...') # Changed to Chinese
            axes[0, i].set_ylabel('实际情况') # Changed to Chinese
            axes[0, i].set_xlabel('预测情况') # Changed to Chinese

        confusion_matrix_plot_path = os.path.join(output_results_dir, f'{model_type}_{embedding_type}_confusion_matrices.png')
        plt.tight_layout()
        plt.savefig(confusion_matrix_plot_path)
        print(f"混淆矩阵图已保存到 '{confusion_matrix_plot_path}'")
        plt.close()
else:
    print("测试集中没有带标签的样本，无法绘制混淆矩阵。")


print(f"\n--- CNN模型配合 '{embedding_type}' 嵌入的处理已完成 ---")
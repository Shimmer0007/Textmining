import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import rcParams

# --- 配置与路径定义 ---
embedding_type = 'one_hot'
model_type = 'cnn'

base_data_path = r'D:\Codes\textmining\data'
classification_data_dir = os.path.join(base_data_path, 'classification_data', embedding_type)
labels_classes_path = os.path.join(base_data_path, 'processed', 'research_areas_classes.pkl')

output_models_dir = os.path.join(base_data_path, '..', 'models') # 模型保存在项目根目录下的models文件夹
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
    print(f"错误: 数据文件未找到 - {e.filename}。请确保已运行数据划分脚本。")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# --- 2. 准备输入数据以适配CNN ---
# Conv1D期望输入形状为 (batch_size, steps, channels)
# 我们将 (num_samples, embedding_dim) 转换为 (num_samples, embedding_dim, 1)
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
    x = Conv1D(128, 5, activation='relu')(input_tensor)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    # 使用GlobalMaxPooling1D替代Flatten，可以直接处理变长序列（尽管这里是固定长度）
    # 并且通常在文本分类的CNN中表现良好
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
                  # F1Score可以通过tf.keras.metrics.F1Score(average='micro', threshold=0.5)添加 (TF 2.x+)

# --- 5. 训练模型 ---
print("\n--- 正在训练模型 ---")
epochs = 30 # 可以根据需要调整
batch_size = 32

# 早停法，防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# 模型检查点，保存效果最好的模型
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

history = cnn_model.fit(
    X_train_reshaped, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# 如果因为EarlyStopping而提前停止，加载最佳权重模型
print(f"\n从 '{model_save_path}' 加载最佳权重模型 (如果 EarlyStopping 生效)...")
cnn_model = load_model(model_save_path) # Keras 会自动处理最佳权重的保存与加载

# --- 6. 评估模型 ---
print("\n--- 正在评估模型 ---")
loss, binary_acc, precision, recall = cnn_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"测试集损失: {loss:.4f}")
print(f"测试集二元准确率 (Keras): {binary_acc:.4f}") # 逐元素准确率
print(f"测试集精确率 (Keras Precision): {precision:.4f}")
print(f"测试集召回率 (Keras Recall): {recall:.4f}")

y_pred_proba = cnn_model.predict(X_test_reshaped)
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# Scikit-learn 指标
subset_acc = accuracy_score(y_test, y_pred_binary) # 精确匹配率
f1_micro = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)
f1_macro = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
f1_samples = f1_score(y_test, y_pred_binary, average='samples', zero_division=0)

print(f"\nScikit-learn 指标:")
print(f"  子集准确率 (Exact Match Ratio): {subset_acc:.4f}")
print(f"  F1 Score (Micro): {f1_micro:.4f}")
print(f"  F1 Score (Macro): {f1_macro:.4f}")
print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
print(f"  F1 Score (Samples): {f1_samples:.4f}")

# --- 7. 保存结果 ---
print("\n--- 正在保存结果 ---")
# 记录到accuracy_scores.txt
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
header = "Timestamp,Embedding,Model,Loss,BinaryAccuracy,Precision,Recall,SubsetAccuracy,F1_Micro,F1_Macro,F1_Weighted,F1_Samples\n"
if not os.path.exists(accuracy_scores_file):
    with open(accuracy_scores_file, 'w') as f:
        f.write(header)

with open(accuracy_scores_file, 'a') as f:
    f.write(f"{timestamp},{embedding_type},{model_type},{loss:.4f},{binary_acc:.4f},{precision:.4f},{recall:.4f},"
            f"{subset_acc:.4f},{f1_micro:.4f},{f1_macro:.4f},{f1_weighted:.4f},{f1_samples:.4f}\n")
print(f"评估结果已追加到 '{accuracy_scores_file}'")

# 可视化训练历史
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 可视化训练历史部分保持不变
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')  # 现在可以正常显示中文
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['binary_accuracy'], label='训练二元准确率')
plt.plot(history.history['val_binary_accuracy'], label='验证二元准确率')
plt.title('二元准确率变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Binary Accuracy')
plt.legend()

history_plot_path = os.path.join(output_results_dir, f'{model_type}_{embedding_type}_training_history.png')
plt.tight_layout()
plt.savefig(history_plot_path)
print(f"训练历史图已保存到 '{history_plot_path}'")
plt.close()

# 混淆矩阵可视化 (示例：为出现最多的3个类别绘制)
print("\n正在生成混淆矩阵图 (示例)...")
mcm = multilabel_confusion_matrix(y_test, y_pred_binary)

# 获取标签出现频率，选择频率最高的几个来展示
all_labels_flat_test = [class_names[i] for i, col_sum in enumerate(y_test.sum(axis=0)) for _ in range(int(col_sum))]
label_counts_test = Counter(all_labels_flat_test)
top_n_classes_to_plot = 3
top_classes_names = [item[0] for item in label_counts_test.most_common(top_n_classes_to_plot)]
class_indices_to_plot = [i for i, name in enumerate(class_names) if name in top_classes_names]

if not class_indices_to_plot: # 如果没有高频类别或class_names有问题
    print("无法确定用于绘制混淆矩阵的高频类别。跳过混淆矩阵绘制。")
else:
    fig, axes = plt.subplots(1, top_n_classes_to_plot, figsize=(5 * top_n_classes_to_plot, 4))
    if top_n_classes_to_plot == 1: # 如果只画一个，axes不是数组
        axes = [axes] 
        
    for i, class_idx in enumerate(class_indices_to_plot):
        if i >= len(axes): break # 确保不会超出axes的范围
        class_name_actual = class_names[class_idx]
        sns.heatmap(mcm[class_idx], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        axes[i].set_title(f'CM for: {class_name_actual[:25]}...') # 截断长名
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

    confusion_matrix_plot_path = os.path.join(output_results_dir, f'{model_type}_{embedding_type}_confusion_matrices.png')
    plt.tight_layout()
    plt.savefig(confusion_matrix_plot_path)
    print(f"混淆矩阵图已保存到 '{confusion_matrix_plot_path}'")
    plt.close()

print(f"\n--- CNN模型配合 '{embedding_type}' 嵌入的处理已完成 ---")

# 在模型评估部分修改如下：

# 尝试调整分类阈值
thresholds = np.linspace(0.1, 0.9, 9)  # 测试不同阈值
for threshold in thresholds:
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    print(f"\nThreshold: {threshold:.2f}")
    print(classification_report(y_test, y_pred_binary, target_names=class_names, zero_division=0))

# 检查数据不平衡情况
print("\n类别分布统计:")
print(pd.DataFrame(y_train.sum(axis=0), index=class_names, columns=['Count']))
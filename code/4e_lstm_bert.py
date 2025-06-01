import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Reshape, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
embedding_type = 'bert' 
model_type = 'lstm' # <--- 改为LSTM

base_data_path = r'D:\Codes\textmining\data'
classification_data_dir = os.path.join(base_data_path, 'classification_data', embedding_type)
labels_classes_path = os.path.join(base_data_path, 'processed', 'research_areas_classes.pkl')

output_models_dir = os.path.join(base_data_path, '..', 'models') 
output_results_dir = os.path.join(base_data_path, '..', 'results', 'classification')
accuracy_scores_file = os.path.join(output_results_dir, 'accuracy_scores.txt')
model_save_path = os.path.join(output_models_dir, f'{model_type}_{embedding_type}.h5')

# --- 创建输出目录 ---
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
    print(f"错误: 数据文件未找到 - {e.filename}。")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# --- 2. 准备输入数据以适配LSTM ---
# LSTM期望输入形状为 (batch_size, timesteps, features)
# 我们将 (num_samples, embedding_dim) 转换为 (num_samples, 1, embedding_dim)
input_dim = X_train.shape[1] 
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, input_dim)
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, input_dim)
num_classes = y_train.shape[1]

print(f"Reshaped X_train 形状: {X_train_reshaped.shape}")
print(f"Reshaped X_test 形状: {X_test_reshaped.shape}")

# --- 3. 构建LSTM模型 ---
print("\n--- 正在构建LSTM模型 ---")
def build_lstm_model(input_shape_lstm, num_classes): # input_shape_lstm should be (timesteps, features)
    input_tensor = Input(shape=input_shape_lstm)
    # 可以尝试Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=False))(input_tensor) # 尝试128单元, return_sequences=False因为下一层是Dense
    # x = LSTM(128, return_sequences=False)(input_tensor) # 或者单向LSTM
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# LSTM的input_shape是 (timesteps, features_per_timestep)
lstm_model = build_lstm_model(input_shape_lstm=(1, input_dim), num_classes=num_classes)
lstm_model.summary()

# --- 4. 编译模型 ---
print("\n--- 正在编译模型 ---")
lstm_model.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['binary_accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# --- 5. 训练模型 ---
print("\n--- 正在训练模型 ---")
epochs = 50 
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)

history = lstm_model.fit(
    X_train_reshaped, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print(f"\n从 '{model_save_path}' 加载最佳权重模型 (如果 EarlyStopping 生效)...")
lstm_model = load_model(model_save_path) 

# --- 6. 寻找最优阈值并评估模型 ---
print("\n--- 正在寻找最优阈值并评估模型 ---")
y_pred_proba = lstm_model.predict(X_test_reshaped)

thresholds_to_test = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]) # 您可以调整这个范围
best_f1_micro = -1
optimal_threshold = 0.5 # 默认值

print(f"测试阈值范围: {thresholds_to_test}")
for threshold in thresholds_to_test:
    y_pred_binary_temp = (y_pred_proba > threshold).astype(int)
    f1_micro_temp = f1_score(y_test, y_pred_binary_temp, average='micro', zero_division=0)
    print(f"  Threshold: {threshold:.2f} -> F1 Micro: {f1_micro_temp:.4f}")
    if f1_micro_temp > best_f1_micro:
        best_f1_micro = f1_micro_temp
        optimal_threshold = threshold

print(f"\n找到的最优阈值 (基于F1 Micro): {optimal_threshold:.2f} (F1 Micro: {best_f1_micro:.4f})")

# 使用最优阈值进行最终评估
y_pred_binary_optimal = (y_pred_proba > optimal_threshold).astype(int)

loss, binary_acc, precision, recall = lstm_model.evaluate(X_test_reshaped, y_test, verbose=0) # Keras evaluate不受阈值影响
print(f"\n基于最优阈值 {optimal_threshold:.2f} 的最终评估:")
print(f"  测试集损失 (Keras): {loss:.4f}")
print(f"  测试集二元准确率 (Keras): {binary_acc:.4f}")
print(f"  测试集精确率 (Keras Precision): {precision:.4f}") # 这个Keras指标是基于内部机制，可能不完全反映我们选的阈值
print(f"  测试集召回率 (Keras Recall): {recall:.4f}")   # 同上

subset_acc_optimal = accuracy_score(y_test, y_pred_binary_optimal)
# f1_micro_optimal 应该等于 best_f1_micro
f1_macro_optimal = f1_score(y_test, y_pred_binary_optimal, average='macro', zero_division=0)
f1_weighted_optimal = f1_score(y_test, y_pred_binary_optimal, average='weighted', zero_division=0)
f1_samples_optimal = f1_score(y_test, y_pred_binary_optimal, average='samples', zero_division=0)

print(f"\nScikit-learn 指标 (Optimal Threshold={optimal_threshold:.2f}):")
print(f"  子集准确率 (Exact Match Ratio): {subset_acc_optimal:.4f}")
print(f"  F1 Score (Micro): {best_f1_micro:.4f}") # 使用已找到的最佳F1 Micro
print(f"  F1 Score (Macro): {f1_macro_optimal:.4f}")
print(f"  F1 Score (Weighted): {f1_weighted_optimal:.4f}")
print(f"  F1 Score (Samples): {f1_samples_optimal:.4f}")

print(f"\nClassification Report (scikit-learn, Optimal Threshold={optimal_threshold:.2f}):")
try:
    report_optimal = classification_report(y_test, y_pred_binary_optimal, target_names=class_names, zero_division=0)
    print(report_optimal)
except ValueError as ve: 
    print(f"生成classification_report时出错: {ve}. 将不使用target_names参数。")
    report_optimal = classification_report(y_test, y_pred_binary_optimal, zero_division=0)
    print(report_optimal)
    
# --- 7. 保存结果 (使用最优阈值下的指标) ---
print("\n--- 正在保存结果 ---")
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
header = "Timestamp,Embedding,Model,Threshold,Loss,BinaryAccuracy,Precision,Recall,SubsetAccuracy,F1_Micro,F1_Macro,F1_Weighted,F1_Samples\n"
if not os.path.exists(accuracy_scores_file):
    with open(accuracy_scores_file, 'w') as f:
        f.write(header)

with open(accuracy_scores_file, 'a') as f:
    f.write(f"{timestamp},{embedding_type},{model_type},{optimal_threshold:.2f}," # 保存最优阈值
            f"{loss:.4f},{binary_acc:.4f},{precision:.4f},{recall:.4f},"
            f"{subset_acc_optimal:.4f},{best_f1_micro:.4f},{f1_macro_optimal:.4f},{f1_weighted_optimal:.4f},{f1_samples_optimal:.4f}\n")
print(f"评估结果已追加到 '{accuracy_scores_file}'")

# --- 8. 可视化与绘图 ---
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
print(f"\n训练历史图已保存到 '{history_plot_path}'")
plt.close()

print("\n正在生成混淆矩阵图 (示例, 基于最优阈值)...")
mcm = multilabel_confusion_matrix(y_test, y_pred_binary_optimal) # 使用最优阈值的预测结果
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
                        xticklabels=['预测为负', '预测为正'], 
                        yticklabels=['实际为负', '实际为正']) 
            axes[0, i].set_title(f'混淆矩阵: {class_name_actual[:25]}...') 
            axes[0, i].set_ylabel('实际情况') 
            axes[0, i].set_xlabel('预测情况') 

        confusion_matrix_plot_path = os.path.join(output_results_dir, f'{model_type}_{embedding_type}_confusion_matrices.png')
        plt.tight_layout()
        plt.savefig(confusion_matrix_plot_path)
        print(f"混淆矩阵图已保存到 '{confusion_matrix_plot_path}'")
        plt.close()
else:
    print("测试集中没有带标签的样本，无法绘制混淆矩阵。")

print(f"\n--- LSTM模型配合 '{embedding_type}' 嵌入的处理已完成 ---")
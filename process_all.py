#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
このスクリプトは、Polisコンサルテーションデータを処理し、
可視化用のウェブサイトを単一のプロセスで生成します。

処理ステップ:
1. CSVデータの読み込み
2. AIプロンプトを用いたデータの処理（抽出、ラベリング、要約）
3. クラスタリング処理
4. 静的なHTML/CSS/JS生成

使用方法:
python process_all.py [--config CONFIG_FILE] [--input INPUT_CSV] [--output OUTPUT_DIR] [--clusters NUM_CLUSTERS] [--api {anthropic,openai}]
"""

import os
import sys
import json
import argparse
import logging
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

# 外部依存ライブラリ
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tttc-processor")

# ------------------------------------------------------------
# ユーティリティ関数
# ------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        sys.exit(1)

def ensure_dir(path: str) -> None:
    """ディレクトリが存在することを確認"""
    os.makedirs(path, exist_ok=True)

def load_prompt(filename: str) -> str:
    """プロンプトファイルを読み込む"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"プロンプトファイル {filename} の読み込みに失敗しました: {e}")
        sys.exit(1)

def save_json(data: Any, filepath: str) -> None:
    """JSONデータを保存する"""
    try:
        # NumPy型をPythonの標準型に変換するためのJSON Encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"JSONデータを {filepath} に保存しました")
    except Exception as e:
        logger.error(f"JSONデータの保存に失敗しました: {e}")

# ------------------------------------------------------------
# API呼び出し関数
# ------------------------------------------------------------

def call_anthropic_api(prompt: str, config: Dict[str, Any]) -> str:
    """Anthropic APIを呼び出す"""
    try:
        # 環境変数からAPIキーを取得（優先）
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # 環境変数にない場合はconfig.jsonから取得
        if not api_key:
            api_key = config["ai"].get("anthropic_api_key") or config["ai"].get("api_key")
        
        if not api_key or api_key == "your_api_key_here" or api_key == "your_anthropic_api_key_here":
            logger.warning("Anthropic APIキーが設定されていません。API呼び出しをスキップします。")
            return "APIキーが設定されていないため、デフォルトの応答を返します。"
        
        # プロンプトからシステムプロンプトとユーザープロンプトを分離
        system_prompt = ""
        user_prompt = prompt
        
        if "/system" in prompt and "/human" in prompt:
            parts = prompt.split("/human", 1)
            if "/system" in parts[0]:
                system_prompt = parts[0].replace("/system", "").strip()
                user_prompt = parts[1].strip()
        
        # Anthropic Python SDKを使用する代わりに、直接APIを呼び出す
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key
        }
        
        # 最新のAnthropicのAPIでは、messagesフォーマットを使用
        # config.jsonからモデル名を取得
        model_name = config["ai"]["model"]
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": config["ai"].get("max_tokens", 4000),
            "temperature": config["ai"].get("temperature", 0.2)
        }
        
        # システムプロンプトがある場合は追加
        if system_prompt:
            data["system"] = system_prompt
        
        retry_attempts = config.get("retry_attempts", 3)
        retry_delay = config.get("retry_delay", 5)
        
        # デバッグ情報を出力
        logger.info(f"Anthropic API リクエスト: モデル={model_name}, システムプロンプト={len(system_prompt) > 0}")
        
        for attempt in range(retry_attempts):
            try:
                # 最新のエンドポイントを使用
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                # エラーレスポンスの詳細を出力
                if response.status_code != 200:
                    logger.error(f"API エラーレスポンス: {response.status_code} {response.text}")
                
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            except Exception as e:
                logger.warning(f"API呼び出しに失敗しました (試行 {attempt+1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("APIリクエストが最大試行回数を超えました")
                    raise
    except Exception as e:
        logger.error(f"Anthropic API呼び出し中にエラーが発生しました: {e}")
        raise

def call_openai_api(prompt: str, config: Dict[str, Any]) -> str:
    """OpenAI APIを呼び出す"""
    # 環境変数からAPIキーを取得（優先）
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # 環境変数にない場合はconfig.jsonから取得
    if not api_key:
        api_key = config["ai"].get("openai_api_key") or config["ai"].get("api_key")
    
    if not api_key or api_key == "your_api_key_here" or api_key == "your_openai_api_key_here":
        logger.error("OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYまたはconfig.jsonのai.openai_api_keyを設定してください。")
        sys.exit(1)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config["ai"].get("model", "gpt-4"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config["ai"].get("max_tokens", 4000),
        "temperature": config["ai"].get("temperature", 0.2)
    }
    
    retry_attempts = config.get("retry_attempts", 3)
    retry_delay = config.get("retry_delay", 5)
    
    for attempt in range(retry_attempts):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"API呼び出しに失敗しました (試行 {attempt+1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                logger.error("APIリクエストが最大試行回数を超えました")
                raise

def call_ai_api(prompt: str, config: Dict[str, Any]) -> str:
    """設定に基づいて適切なAI APIを呼び出す"""
    provider = config["ai"]["provider"].lower()
    
    if provider == "anthropic":
        return call_anthropic_api(prompt, config)
    elif provider == "openai":
        return call_openai_api(prompt, config)
    else:
        logger.error(f"サポートされていないAPIプロバイダー: {provider}")
        sys.exit(1)

# ------------------------------------------------------------
# データ処理関数
# ------------------------------------------------------------

def extract_arguments(comments_df: pd.DataFrame, extraction_prompt: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """コメントから意見を抽出する"""
    comment_column = config["input"]["comment_column"]
    comments = comments_df[comment_column].dropna().tolist()
    
    logger.info(f"{len(comments)}件のコメントから意見を抽出します")
    arguments = []
    
    # バッチ処理のサイズ
    batch_size = config["ai"].get("sample_size", 5)
    
    # プログレスバー付きでバッチ処理
    for i in tqdm(range(0, len(comments), batch_size), desc="意見抽出"):
        batch = comments[i:i+batch_size]
        
        # 各バッチのプロンプトを準備
        batch_text = "\n\n".join([f"/human\n\n{comment}" for comment in batch])
        prompt = extraction_prompt + batch_text
        
        try:
            # AI APIを呼び出し
            response = call_ai_api(prompt, config)
            
            # レスポンスをJSONとして解析
            # 各コメントに対する抽出された意見のリストを取得
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, response, re.DOTALL)
            
            for j, match in enumerate(matches):
                if i + j < len(batch):  # インデックスが範囲内であることを確認
                    # JSONとして解析
                    try:
                        # 角括弧を追加して有効なJSONにする
                        json_str = f"[{match}]"
                        extracted_args = json.loads(json_str)
                        
                        for arg in extracted_args:
                            arguments.append({
                                "original_comment": batch[j],
                                "argument": arg,
                                "comment_id": comments_df.iloc[i+j].get("comment-id", i+j) if i+j < len(comments_df) else i+j
                            })
                    except json.JSONDecodeError:
                        # JSON解析に失敗した場合は、テキストとして処理
                        arguments.append({
                            "original_comment": batch[j],
                            "argument": match.strip('"\'').strip(),
                            "comment_id": comments_df.iloc[i+j].get("comment-id", i+j) if i+j < len(comments_df) else i+j
                        })
            
            # APIレート制限を考慮して少し待機
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"バッチ {i//batch_size + 1} の処理中にエラーが発生しました: {e}")
            # エラーが発生しても処理を続行
            continue
    
    logger.info(f"{len(arguments)}件の意見を抽出しました")
    return arguments

def cluster_arguments(arguments: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """意見をクラスタリングする"""
    # 意見のテキストを取得
    texts = [arg["argument"] for arg in arguments]
    
    # TF-IDFベクトル化
    logger.info("テキストをベクトル化しています...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # K-meansクラスタリング
    n_clusters = min(config["clustering"]["num_clusters"], len(texts))
    logger.info(f"{len(texts)}件の意見を{n_clusters}クラスターに分類します")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # PCAで2次元に削減（可視化用）
    logger.info("2次元に次元削減しています...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())
    
    # 結果をまとめる
    for i, arg in enumerate(arguments):
        arg["cluster"] = int(clusters[i])
        arg["x"] = float(coords[i, 0])
        arg["y"] = float(coords[i, 1])
    
    return arguments, clusters, coords

def generate_cluster_labels(clustered_arguments: List[Dict[str, Any]], labelling_prompt: str, config: Dict[str, Any]) -> List[str]:
    """クラスターにラベルを付ける"""
    n_clusters = config["clustering"]["num_clusters"]
    logger.info(f"{n_clusters}個のクラスターにラベルを付けます")
    
    labels = []
    
    # 各クラスターに対してラベルを生成
    for cluster_id in range(n_clusters):
        # クラスター内の意見を抽出
        cluster_args = [arg["argument"] for arg in clustered_arguments if arg["cluster"] == cluster_id]
        
        if not cluster_args:
            labels.append(f"クラスター {cluster_id + 1}")
            continue
        
        # サンプルとして最大10件を使用
        sample_size = min(len(cluster_args), config["ai"].get("sample_size", 5))
        sample_args = cluster_args[:sample_size]
        
        # クラスター外の意見をサンプリング
        other_args = [arg["argument"] for arg in clustered_arguments if arg["cluster"] != cluster_id]
        other_sample = np.random.choice(other_args, min(len(other_args), sample_size), replace=False).tolist()
        
        # プロンプトの準備
        prompt = labelling_prompt
        prompt += f"\n\n/human\n\nコンサルテーションの質問：「{config['content']['question']}」\n\n"
        prompt += "関心のあるクラスター以外の議論の例：\n\n"
        prompt += "\n".join([f"* {arg}" for arg in other_sample])
        prompt += "\n\nクラスター内の議論の例：\n\n"
        prompt += "\n".join([f"* {arg}" for arg in sample_args])
        
        try:
            # AI APIを呼び出し
            response = call_ai_api(prompt, config)
            labels.append(response.strip())
            
            # APIレート制限を考慮して少し待機
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"クラスター {cluster_id + 1} のラベル生成中にエラーが発生しました: {e}")
            labels.append(f"クラスター {cluster_id + 1}")
    
    return labels

def generate_cluster_summaries(clustered_arguments: List[Dict[str, Any]], labels: List[str], takeaways_prompt: str, config: Dict[str, Any]) -> List[str]:
    """クラスターの要約を生成する"""
    n_clusters = config["clustering"]["num_clusters"]
    logger.info(f"{n_clusters}個のクラスターの要約を生成します")
    
    summaries = []
    
    # 各クラスターに対して要約を生成
    for cluster_id in range(n_clusters):
        # クラスター内の意見を抽出
        cluster_args = [arg["argument"] for arg in clustered_arguments if arg["cluster"] == cluster_id]
        
        if not cluster_args:
            summaries.append(f"このクラスターには意見がありません。")
            continue
        
        # プロンプトの準備
        prompt = takeaways_prompt
        prompt += f"\n\n/human\n\n[\n"
        prompt += ",\n".join([f'\t"{arg}"' for arg in cluster_args[:20]])  # 最大20件を使用
        prompt += "\n]"
        
        try:
            # AI APIを呼び出し
            response = call_ai_api(prompt, config)
            summaries.append(response.strip())
            
            # APIレート制限を考慮して少し待機
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"クラスター {cluster_id + 1} の要約生成中にエラーが発生しました: {e}")
            summaries.append(f"このクラスターの要約を生成できませんでした。")
    
    return summaries

def generate_overview(labels: List[str], summaries: List[str], arguments: List[Dict[str, Any]], comments_df: pd.DataFrame, overview_prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """全体の概要を生成する"""
    # 作品ごとにデータをグループ化
    works = sorted(comments_df["作品"].unique())
    
    # 各作品のクラスター情報を収集
    work_clusters = {}
    for work in works:
        work_clusters[work] = []
        
    # 各クラスターを対応する作品に割り当て
    for i, (label, summary) in enumerate(zip(labels, summaries)):
        # クラスターに属する意見を取得
        cluster_args = [arg for arg in arguments if arg["cluster"] == i]
        
        # この意見がどの作品に属するかを確認
        for arg in cluster_args:
            orig_comment = arg["original_comment"]
            # コメントに対応する作品を探す
            works_for_comment = comments_df[comments_df[config["input"]["comment_column"]] == orig_comment]["作品"].tolist()
            if works_for_comment:
                work = works_for_comment[0]
                if not any(item["label"] == label for item in work_clusters[work]):
                    work_clusters[work].append({
                        "label": label,
                        "summary": summary,
                        "arguments": []
                    })
                
                # 対応するクラスター情報に意見を追加
                for item in work_clusters[work]:
                    if item["label"] == label:
                        item["arguments"].append(arg["argument"])
    
    # プロンプトの準備
    prompt = overview_prompt
    prompt += f"\n\n/human\n\n"
    prompt += f"コンサルテーションの質問：「{config['content']['question']}」\n\n"
    prompt += "作品ごとのクラスター分析：\n\n"
    
    for work, clusters in work_clusters.items():
        prompt += f"### 作品: {work}\n\n"
        for cluster in clusters:
            prompt += f"クラスター: {cluster['label']}\n"
            prompt += f"要約: {cluster['summary']}\n"
            prompt += "意見例:\n"
            for arg in cluster["arguments"][:5]:  # 最大5つの意見を表示
                prompt += f"- {arg}\n"
            prompt += "\n"
        prompt += "\n"
    
    try:
        # AI APIを呼び出し
        response = call_ai_api(prompt, config)
        
        # JSON応答を解析
        try:
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                raise ValueError("JSON形式の応答が見つかりませんでした")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON解析エラー: {e}")
            return {
                "general_overview": "データの解析に問題が発生しました。",
                "categories": []
            }
    except Exception as e:
        logger.error(f"全体の概要生成中にエラーが発生しました: {e}")
        return {
            "general_overview": "全体の概要を生成できませんでした。",
            "categories": []
        }
    
# ------------------------------------------------------------
# ウェブサイト生成関数
# ------------------------------------------------------------

def generate_html(data: Dict[str, Any], comments_df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """HTMLファイルを生成する"""
    arguments = data["arguments"]
    labels = data["labels"]
    summaries = data["summaries"]
    overview_data = data["overview"]
    
    # クラスターごとの色を設定
    colors = config["visualization"]["colors"]
    n_clusters = len(labels)
    
    # 各クラスターの代表的な意見を選択（各クラスターから最大5件）
    representative_args = []
    for cluster_id in range(n_clusters):
        cluster_args = [arg for arg in arguments if arg["cluster"] == cluster_id]
        if cluster_args:
            representative_args.extend(cluster_args[:5])
    
    # 全体の概要
    general_overview = overview_data.get('general_overview', "全体の概要を生成できませんでした。")
    
    # 作品別分析
    categories = overview_data.get('categories', [])
    
    
    # HTMLテンプレート
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config['content']['title']}</title>
    <link rel="stylesheet" href="styles.css">
    <script src="{config['visualization']['plotly_url']}"></script>
</head>
<body>
    <header>
        <h1>{config['content']['title']}</h1>
        <p class="question">{config['content']['question']}</p>
        <p class="description">{config['content']['description']}</p>
    </header>
    
    <main>
        <section class="overview">
            <h2>全体の概要</h2>
            <p>{general_overview}</p>
        </section>
        
        <section class="visualization">
            <h2>意見の分布</h2>
            <div id="scatter-plot"></div>
        </section>
"""

    # 作品別のセクションを追加
    if categories:
        html += """
        <section class="categories">
            <h2>AIが推察する作品分析</h2>
        """
        
        for category in categories:
            html += f"""
            <div class="category-analysis">
                <h3>{category['title']}</h3>
            """
            
            for detail in category.get('detail', []):
                html += f"""
                <div class="category-section">
                    <h4>作品の概要</h4>
                    <p>{detail.get('overview', '')}</p>
                </div>
                
                <div class="category-section">
                    <h4>作品の内容（※AIが作品の劇評を元に推察した内容です。必ずしも実際の劇の内容とは一致しません）</h4>
                    <p>{detail.get('content', '')}</p>
                </div>
                
                <div class="category-section">
                    <h4>意見のつながり</h4>
                    <p>{detail.get('link', '')}</p>
                </div>
                
                <div class="category-section">
                    <h4>良かった点</h4>
                    <p>{detail.get('good', '')}</p>
                </div>
                
                <div class="category-section">
                    <h4>問題点と解決策</h4>
                    <p>{detail.get('bad', '')}</p>
                </div>
                """
            
            html += """
            </div>
            """
        
        html += """
        </section>
        """

    html += f"""
        
        <section class="clusters">
            <h2>クラスター分析</h2>
            <div class="cluster-grid">
"""
    
    # クラスター情報を追加
    for i, (label, summary) in enumerate(zip(labels, summaries)):
        color = colors[i % len(colors)]
        html += f"""
                <div class="cluster-card" style="border-color: {color}">
                    <h3 style="background-color: {color}">{label}</h3>
                    <p class="summary">{summary}</p>
                    <div class="arguments">
                        <h4>代表的な意見:</h4>
                        <ul>
"""
        
        # クラスター内の代表的な意見を追加
        cluster_args = [arg["argument"] for arg in arguments if arg["cluster"] == i]
        for arg in cluster_args[:5]:  # 最大5件表示
            html += f"""
                            <li>{arg}</li>
"""
        
        html += """
                        </ul>
                    </div>
                </div>
"""
    
    # HTMLの残りの部分
    html += """
            </div>
        </section>
        
        <section class="all-comments">
            <h2>全ての意見</h2>
            <table class="comments-table">
                <thead>
                    <tr>
                        <th>作品</th>
                        <th>タイムスタンプ</th>
                        <th>お名前</th>
                        <th>コメント</th>
                    </tr>
                </thead>
                <tbody>
"""

    # CSVデータから全ての意見を追加
    for _, row in comments_df.iterrows():
        # タイムスタンプ、お名前、作品を取得
        timestamp = row["タイムスタンプ"]
        name = row["お名前"]
        category = row["作品"]
        
        # コメント本文
        comment = row[config['input']['comment_column']]
        
        html += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{timestamp}</td>
                        <td>{name}</td>
                        <td>{comment}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </section>
    </main>
    
    <footer>
        <p>生成日時: """ + datetime.now().strftime("%Y年%m月%d日 %H:%M") + """</p>
        <p>&copy;劇団 野らぼう</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>
"""
    
    return html

def generate_css() -> str:
    """CSSファイルを生成する"""
    css = """/* 全体のスタイル */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* ヘッダー */
header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    color: #2c3e50;
}

.question {
    font-size: 1.5rem;
    font-weight: 500;
    margin-bottom: 15px;
    color: #3498db;
}

.description {
    font-size: 1rem;
    color: #7f8c8d;
}

/* メインコンテンツ */
main {
    display: flex;
    flex-direction: column;
    gap: 40px;
}

section {
    background-color: #fff;
    border-radius: 8px;
    padding: 25px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h2 {
    font-size: 1.8rem;
    margin-bottom: 20px;
    color: #2c3e50;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 10px;
}

h3 {
    font-size: 1.5rem;
    margin-bottom: 15px;
    color: #2c3e50;
}

h4 {
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: #2c3e50;
}

/* 全体概要 */
.overview p {
    font-size: 1.1rem;
    line-height: 1.7;
}

/* 可視化 */
.visualization {
    min-height: 500px;
}

#scatter-plot {
    width: 100%;
    height: 500px;
}

/* 作品別分析 */
.categories {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.category-analysis {
    margin-bottom: 30px;
    padding: 20px;
    border: 1px solid #ecf0f1;
    border-radius: 8px;
    background-color: #f9f9f9;
}

.category-section {
    margin-bottom: 20px;
}

.category-section p {
    font-size: 1rem;
    line-height: 1.6;
}

/* クラスター分析 */
.cluster-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 25px;
}

.cluster-card {
    border: 2px solid;
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.3s ease;
}

.cluster-card:hover {
    transform: translateY(-5px);
}

.cluster-card h3 {
    padding: 15px;
    color: white;
    font-size: 1.3rem;
}

.summary {
    padding: 15px;
    font-size: 1rem;
    border-bottom: 1px solid #ecf0f1;
}

.arguments {
    padding: 15px;
}

.arguments h4 {
    margin-bottom: 10px;
    font-size: 1.1rem;
    color: #2c3e50;
}

.arguments ul {
    list-style-position: inside;
    padding-left: 10px;
}

.arguments li {
    margin-bottom: 8px;
    font-size: 0.95rem;
}

/* 全ての意見 */
.comments-table {
    width: 100%;
    border-collapse: collapse;
}

.comments-table th,
.comments-table td {
    padding: 10px;
    border: 1px solid #ecf0f1;
    text-align: left;
}

.comments-table th {
    background-color: #f2f2f2;
    font-weight: bold;
}

.comments-table tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* フッター */
footer {
    margin-top: 40px;
    text-align: center;
    padding: 20px;
    color: #7f8c8d;
    font-size: 0.9rem;
}

/* レスポンシブデザイン */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .question {
        font-size: 1.3rem;
    }
    
    .cluster-grid {
        grid-template-columns: 1fr;
    }
    
    .comments-table {
        font-size: 0.9rem;
    }
    
    .comments-table th,
    .comments-table td {
        padding: 8px;
    }
}
"""
    
    return css

def generate_js(data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """JavaScriptファイルを生成する"""
    arguments = data["arguments"]
    labels = data["labels"]
    
    # クラスターごとの色を設定
    colors = config["visualization"]["colors"]
    n_clusters = len(labels)
    
    # 散布図用のデータを準備
    scatter_data = []
    
    for cluster_id in range(n_clusters):
        cluster_args = [arg for arg in arguments if arg["cluster"] == cluster_id]
        
        if not cluster_args:
            continue
        
        x_values = [arg["x"] for arg in cluster_args]
        y_values = [arg["y"] for arg in cluster_args]
        text_values = [arg["argument"] for arg in cluster_args]
        
        scatter_data.append({
            "cluster": cluster_id,
            "label": labels[cluster_id],
            "color": colors[cluster_id % len(colors)],
            "x": x_values,
            "y": y_values,
            "text": text_values
        })
    
    # JavaScriptコード
    js = """// 散布図の描画
document.addEventListener('DOMContentLoaded', function() {
    const scatterData = """ + json.dumps(scatter_data, ensure_ascii=False) + """;
    
    const traces = scatterData.map(cluster => {
        return {
            x: cluster.x,
            y: cluster.y,
            mode: 'markers',
            type: 'scatter',
            name: cluster.label,
            text: cluster.text,
            hoverinfo: 'text',
            marker: {
                size: 10,
                color: cluster.color,
                opacity: 0.7
            }
        };
    });
    
    const layout = {
        title: '意見の分布',
        hovermode: 'closest',
        xaxis: {
            title: 'X(意見の主要な特徴の違い - 肯定的/批判的、具体的/抽象的など)',
            zeroline: false,
            showgrid: false
        },
        yaxis: {
            title: 'Y(意見の二次的な特徴の違い - 個人的/一般的、感情的/分析的など)',
            zeroline: false,
            showgrid: false
        },
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 80,
            pad: 4
        },
        legend: {
            x: 0,
            y: 1,
            traceorder: 'normal',
            font: {
                family: 'sans-serif',
                size: 12,
                color: '#000'
            },
            bgcolor: '#E2E2E2',
            bordercolor: '#FFFFFF',
            borderwidth: 2
        }
    };
    
    Plotly.newPlot('scatter-plot', traces, layout, {responsive: true});
});

// クラスターカードのインタラクション
document.addEventListener('DOMContentLoaded', function() {
    const clusterCards = document.querySelectorAll('.cluster-card');
    
    clusterCards.forEach(card => {
        card.addEventListener('click', function() {
            // カードがクリックされたときの処理（必要に応じて）
        });
    });
});
"""
    
    return js

def generate_website(data: Dict[str, Any], comments_df: pd.DataFrame, output_dir: str, config: Dict[str, Any]) -> None:
    """ウェブサイトを生成する"""
    logger.info(f"ウェブサイトを {output_dir} に生成しています...")
    
    # 出力ディレクトリを作成
    ensure_dir(output_dir)
    
    # HTMLファイルを生成
    html_content = generate_html(data, comments_df, config)
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # CSSファイルを生成
    css_content = generate_css()
    with open(os.path.join(output_dir, "styles.css"), "w", encoding="utf-8") as f:
        f.write(css_content)
    
    # JavaScriptファイルを生成
    js_content = generate_js(data, config)
    with open(os.path.join(output_dir, "script.js"), "w", encoding="utf-8") as f:
        f.write(js_content)
    
    # データをJSONとして保存（オプション）
    save_json(data, os.path.join(output_dir, "data.json"))
    
    logger.info(f"ウェブサイトの生成が完了しました: {output_dir}")

# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------

def main():
    """メイン処理関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='データ処理と可視化')
    parser.add_argument('--config', type=str, default='config.json', help='設定ファイルのパス')
    parser.add_argument('--input', type=str, help='入力CSVファイルのパス')
    parser.add_argument('--output', type=str, help='出力ディレクトリのパス')
    parser.add_argument('--clusters', type=int, help='クラスター数（設定ファイルの値を上書き）')
    parser.add_argument('--api', type=str, choices=['anthropic', 'openai'], help='使用するAPI（設定ファイルの値を上書き）')
    parser.add_argument('--skip-extraction', action='store_true', help='意見抽出をスキップ')
    parser.add_argument('--skip-clustering', action='store_true', help='クラスタリングをスキップ')
    parser.add_argument('--skip-labelling', action='store_true', help='ラベリングをスキップ')
    parser.add_argument('--skip-summary', action='store_true', help='要約生成をスキップ')
    args = parser.parse_args()
    
    # 処理開始
    start_time = datetime.now()
    logger.info("処理を開始します")
    
    try:
        # 設定の読み込み
        config = load_config(args.config)
        
        # コマンドライン引数で設定を上書き
        if args.input:
            config['input']['file'] = args.input
        if args.output:
            config['output']['dir'] = args.output
        if args.clusters:
            config['clustering']['num_clusters'] = args.clusters
        if args.api:
            config['ai']['provider'] = args.api
        
        # 入力ファイルと出力ディレクトリのパスを設定
        input_file = config['input']['file']
        output_dir = config['output']['dir']
        
        # 出力ディレクトリの準備
        ensure_dir(output_dir)
        
        # ステップ1: CSVデータの読み込み
        logger.info(f"CSVデータを読み込んでいます: {input_file}")
        try:
            comments_df = pd.read_csv(input_file)
            logger.info(f"{len(comments_df)}件のコメントを読み込みました")
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました: {e}")
            sys.exit(1)
        
        # プロンプトファイルの読み込み
        prompts_dir = config['prompts']['dir']
        extraction_prompt = load_prompt(os.path.join(prompts_dir, config['prompts']['extraction']))
        labelling_prompt = load_prompt(os.path.join(prompts_dir, config['prompts']['labelling']))
        takeaways_prompt = load_prompt(os.path.join(prompts_dir, config['prompts']['takeaways']))
        overview_prompt = load_prompt(os.path.join(prompts_dir, config['prompts']['overview']))
        
        # ステップ2: 意見の抽出
        if not args.skip_extraction:
            logger.info("意見を抽出しています...")
            arguments = extract_arguments(comments_df, extraction_prompt, config)
        else:
            logger.info("意見抽出をスキップします")
            # スキップする場合は、ダミーデータを作成
            arguments = [{"argument": row[config['input']['comment_column']], "original_comment": row[config['input']['comment_column']], "comment_id": i} 
                         for i, row in comments_df.iterrows()]
        
        # ステップ3: クラスタリング
        if not args.skip_clustering:
            logger.info("クラスタリングを実行しています...")
            arguments, clusters, coords = cluster_arguments(arguments, config)
        else:
            logger.info("クラスタリングをスキップします")
            # スキップする場合は、ランダムにクラスターを割り当て
            n_clusters = config['clustering']['num_clusters']
            for arg in arguments:
                arg["cluster"] = np.random.randint(0, n_clusters)
                arg["x"] = np.random.random() * 2 - 1
                arg["y"] = np.random.random() * 2 - 1
        
        # ステップ4: クラスターのラベル付け
        if not args.skip_labelling:
            logger.info("クラスターにラベルを付けています...")
            labels = generate_cluster_labels(arguments, labelling_prompt, config)
        else:
            logger.info("ラベリングをスキップします")
            # スキップする場合は、デフォルトのラベルを使用
            n_clusters = config['clustering']['num_clusters']
            labels = [f"クラスター {i+1}" for i in range(n_clusters)]
        
        # ステップ5: クラスターの要約
        if not args.skip_summary:
            logger.info("クラスターを要約しています...")
            summaries = generate_cluster_summaries(arguments, labels, takeaways_prompt, config)
            
            # ステップ6: 全体の概要生成
            logger.info("全体の概要を生成しています...")
            overview_data = generate_overview(labels, summaries, arguments, comments_df, overview_prompt, config)
        else:
            logger.info("要約生成をスキップします")
            # スキップする場合は、デフォルトの要約を使用
            n_clusters = config['clustering']['num_clusters']
            summaries = [f"クラスター {i+1} の要約" for i in range(n_clusters)]
            overview_data = {
                'general_overview': "全体の概要を生成できませんでした。",
                'categories': []
            }
        
        # ステップ7: ウェブサイト生成
        logger.info("ウェブサイトを生成しています...")
        data = {
            "arguments": arguments,
            "labels": labels,
            "summaries": summaries,
            "overview": overview_data
        }
        generate_website(data, comments_df, output_dir, config)
        
        # 処理完了
        end_time = datetime.now()
        logger.info(f"処理が完了しました。所要時間: {end_time - start_time}")
        logger.info(f"出力先: {output_dir}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

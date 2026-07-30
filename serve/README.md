# serve — 本機推論服務

FastAPI + 單頁前端，用來切換不同訓練結果的 checkpoint 做試聽比對。

```bash
cd serve
cp .env.example .env
docker compose up --build
# http://127.0.0.1:8080
```

映像重用 repo 根目錄 `Dockerfile` 的 `runner` stage，不需要第二份建置設定；FastAPI／uvicorn／python-multipart 已隨 `webui` extra 的 gradio 一併安裝。

## 目錄對應

| 主機路徑 | 容器內 | 用途 |
|---|---|---|
| `../checkpoints` | `/app/checkpoints` | 官方權重（`config.yaml`、`gpt.pth`、`s2mel.pth`、`bpe.model`…） |
| `../models` | `/app/models` | **把微調好的 `.pth` 丟這裡**，會出現在模型下拉選單 |
| `../tokenizers` | `/app/tokenizers` | 擴充過的 tokenizer（`*.model`），有才需要 |
| `./voices` | `/app/voices` | 常用參考聲音 |
| `./outputs` | `/app/outputs` | 生成結果 |
| `./app` | `/app/serve/app` | 服務程式碼，改了不用重建 |

微調 checkpoint 請先用 `tools/prune_gpt_checkpoint.py` 去掉 optimizer 狀態（7.7 GB → 純權重），否則載入很慢也很吃記憶體。

## 重要：`INDEXTTS_ZH_T2S=1`

`compose.yaml` 已經設好。前處理階段所有文字都經過繁→簡折疊，推論必須一致，否則模型會收到訓練時沒見過的文字分佈。前端狀態列會顯示 `T2S on`；如果顯示紅色的 `off`，先別採信任何試聽結果。

## 輸入是華語漢字

輸入 `你今天吃飽了嗎`，不是 `你今仔日食飽未`。台語模型學的是「華語字形 → 台語發音」的映射，輸入端不需要台文。

## API

```bash
# 模型與 tokenizer 清單
curl localhost:8080/api/models

# 用清單裡的參考聲音合成
curl -X POST localhost:8080/api/tts \
  -F text=你今天吃飽了嗎 \
  -F model=tai8_step23560.pth \
  -F voice=sample_prompt.wav

# 直接上傳參考聲音（只用於這次請求）
curl -X POST localhost:8080/api/tts \
  -F text=你今天吃飽了嗎 \
  -F reference=@my_voice.wav \
  -o out.json

# 存為常用聲音
curl -X POST localhost:8080/api/voices -F file=@my_voice.wav
```

| 端點 | 說明 |
|---|---|
| `GET /api/health` | GPU 記憶體、目前載入的模型、T2S 狀態 |
| `GET /api/models` | 可選的 checkpoint 與 tokenizer |
| `GET /api/voices` · `POST` · `DELETE /api/voices/{id}` | 常用參考聲音 |
| `POST /api/tts` | 合成，回傳 JSON 含 `url` |
| `GET /api/audio/{name}` | 取回音檔 |
| `POST /api/unload` | 釋放 VRAM |

`POST /api/tts` 可調參數：`temperature`、`top_p`、`top_k`、`repetition_penalty`、`num_beams`、`max_mel_tokens`、`max_text_tokens_per_sentence`、`interval_silence`、`seed`、`emo_text`、`emo_alpha`。

沒有開放 `duration_seconds`：`infer_v2_modded.py` 的換算率是 `24000/1024 = 23.4` tokens/秒，但實測 semantic code 是 50 Hz，指定秒數會得到約一半的長度。修好之前不提供。

## 一次只載入一個模型

單張卡放不下兩份 1.5B 權重，所以切換模型會先卸載前一個。切換後第一次請求要等權重載入（約一分鐘）。合成請求以 lock 串行化，並發請求會排隊而不是搶 GPU。

需要多少 VRAM：fp16 約 6–8 GB。本機若被其他 process 佔用，先看 `GET /api/health` 的 `free_mb`。

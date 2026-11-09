# Предобучение S AI LLM 1.0

[Отчёт и результаты](REPORT.md). Исходный `CopyLLM` содержит 1 033 003 011 параметров. Файлы весов загружаются с `weights_only=True`.

## Colab

Откройте [ноутбук T4](S_AI_LLM_1_0_Pretraining_T4.ipynb), выберите Tesla T4 и выполняйте ячейки по порядку. В режиме отдельных файлов загрузите в `/content` исходный `s_ai_llm_1_0_untrained_expanded.pt` и, при наличии, `sai1b_wikipedia_pilot_10m.tar.gz`. Папка результатов — `/content/sai1b-results`. Скачайте состояние и метрики до завершения сессии. Для работы через Google Drive включите `USE_DRIVE` в первой кодовой ячейке. Тогда результаты будут в `MyDrive/S_AI_Training/S_AI_LLM_1_0/pretraining_v3_20261108`.

Пилот сохраняет состояние после шага 12 и продолжает до шага 24. Для возобновления в новом сеансе выполните ячейки настройки и корпуса, затем ячейку возобновления. В режиме отдельных файлов сначала верните сохранённый `training-state.pt` в `/content/sai1b-results/t4-pilot`.

## Продолжение после шага 24 на T4

[Профиль продолжения](t4_continuation.json) задаёт 512 новых шагов, 524 288 позиций поровну на RU и EN и обновление всех 20 блоков. Состав validation и test тот же, что в пилоте при использовании того же корпуса. Новый seed меняет последовательность тренировочных окон. Новый этап начинает собственное расписание скорости обучения и сохраняет состояние каждые 128 шагов.

В Colab выполните ячейки настройки и подготовки корпуса. В терминале перейдите в корень актуальной ветки репозитория. Для `PILOT` используйте локальную копию `t4-pilot/training-state.pt`, для `CORPUS` — каталог распакованного пилотного корпуса, для `RUN` — новый каталог на диске с сохранением между сеансами.

```bash
PILOT=/path/to/local/t4-step24.pt
CORPUS=/path/to/wiki-pilot
RUN=/path/to/persistent/t4-continuation
printf '0f5f6547d27e1482571f8c8b5defae08b14da072d2dfa720b0c049100cce9044  %s\n' "$PILOT" | sha256sum --check &&
  python -m training_1b.pretrain --profile training_1b/t4_continuation.json \
  --corpus "$CORPUS" --init "$PILOT" --output "$RUN" --stop-after 128
```

После остановки на шаге 128 или прерывания возьмите локальную копию последнего `training-state.pt` из нового каталога. Для возобновления используйте тот же профиль, корпус и версии среды.

```bash
RESUME=/path/to/local/t4-continuation-state.pt
python -m training_1b.pretrain --profile training_1b/t4_continuation.json \
  --corpus "$CORPUS" --resume "$RESUME" --output "$RUN"

FINAL=/path/to/local/t4-continuation-final.pt
python -m training_1b.evaluate --checkpoint "$FINAL" \
  --output "$RUN/answers.json" --device cuda --precision fp16 --max-tokens 32
```

Повторите проверку с `--vocabulary-only --output "$RUN/answers-vocabulary.json"`, чтобы отдельно увидеть, что даёт обученная словарная проекция. Обычный вызов оценивает штатную генерацию CopyLLM.

## Проверочный запуск на Selectel PRO 6000 96 ГБ

На уже запущенном сервере Linux перенесите в закрытый каталог чекпойнт шага 24 и архив `sai1b_wikipedia_pilot_10m.tar.gz`. Убедитесь, что драйвер видит GPU, и установите PyTorch с CUDA 12.8 в отдельное виртуальное окружение. [GPU в Selectel](https://docs.selectel.ru/cloud-servers/create/gpus/) · [Сборки PyTorch](https://pytorch.org/get-started/previous-versions/)

```bash
nvidia-smi
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install tokenizers==0.21.4 filelock
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(including_emulation=False)
sample = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
assert (sample @ sample)[0, 0].item() == 256
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
PY
```

Из корня рабочей ветки укажите проверенный локальный файл `PILOT`, распакованный `CORPUS` и новый постоянный каталог `RUN` по образцу выше. Начните с 16 шагов BF16 на существующем профиле. Это 1 048 576 позиций с записью состояния в конце отрезка.

```bash
printf '0f5f6547d27e1482571f8c8b5defae08b14da072d2dfa720b0c049100cce9044  %s\n' "$PILOT" | sha256sum --check &&
  python -m training_1b.pretrain --profile training_1b/a100_h100.json \
    --corpus "$CORPUS" --init "$PILOT" --output "$RUN" --stop-after 16
```

По `metrics.json` проверьте токены в секунду, время шагов, пик памяти и validation loss обоих языков. Для повторного запуска с тем же профилем и корпусом используйте `--resume "$RUN/training-state.pt"` и более поздний `--stop-after`. При переходе на больший корпус начните отдельный этап через `--init`.

## A100 или H100

```bash
git clone --branch codex/sai-llm-1b-pretraining https://github.com/thepeshkov7/S-AI-LLM.git
cd S-AI-LLM
python -m pip install -r requirements.txt filelock pyarrow==20.0.0
BASE=/path/to/s_ai_llm_1_0_untrained_expanded.pt

python -m training_1b.corpus --checkpoint "$BASE" --cache corpus/cache \
  --output corpus/wiki-100m --wiki-shards 2 --limit-tokens 50000000

python -m training_1b.pretrain --profile training_1b/a100_h100.json \
  --corpus corpus/wiki-100m --init "$BASE" --output runs/a100 --stop-after 200

python -m training_1b.pretrain --profile training_1b/a100_h100.json \
  --corpus corpus/wiki-100m --resume runs/a100/training-state.pt \
  --output runs/a100 --stop-after 500
```

Измерьте `train_tokens_per_second`, `peak_vram_allocated_gib` и validation каждого языка. Возобновление сверяет корпус, профиль, версии Torch и оптимизатора, аппаратную платформу и сохранённое состояние. Для новой фазы с другим профилем используйте `--init` и отдельный каталог. Полное состояние 1B с AdamW может занять около 11,6 ГиБ. При атомарной записи понадобится место для двух состояний.

Для T4 используется `t4_pilot.json` и `bitsandbytes==0.48.2`. `last_blocks` определяет число размороженных блоков. `objective: copy` включает градиенты указателя копирования. Полный срез Wikipedia доступен через `--wiki-shards 0 --limit-tokens 0`.

Корпус диалогов и задач можно воспроизвести командой `python -m training_1b.instructions --checkpoint "$BASE" --cache corpus/cache --output corpus/instructions`. Перед дальнейшим обучением используйте результат пересборки. Сборщик исключает шаблон `pencil-boxes-3-4-13` с некорректным числом использованных карандашей, попавший в первый архив.

## Восстановление CPU-пилота

На момент публикации в Drive были доступны 12 из 14 частей. Для восстановления с имеющимися файлами заново воспроизведите CPU-пилот по профилю ниже. Когда доступны все части, сложите все 14 частей с номерами 00–13 в один каталог и объедините их. Проверьте SHA-256 полученного файла по [метрикам](results/cpu-pilot.json).

```bash
cat sai1b_cpu_step8.pt.part-* > sai1b_cpu_step8.pt
sha256sum sai1b_cpu_step8.pt
```

Из корня репозитория восстановите файл из исходных весов и компактного состояния.

```bash
python -m training_1b.portable restore --base "$BASE" \
  --input /path/to/sai1b_cpu_step8.pt --output runs/cpu-step8-restored.pt
python -m training_1b.evaluate --checkpoint runs/cpu-step8-restored.pt \
  --output runs/cpu-step8-answers.json --device cpu --precision bf16 --max-tokens 24
```

Для воспроизведения исходного CPU-пилота возьмите `cpu_pilot.json`, `--device cpu` и `--stop-after 4`, затем выполните `--resume` с тем же профилем. Для нового этапа после шага 8 используйте `--init` и отдельный профиль.

При распространении данных Wikipedia сохраняйте `source-manifest.json` и `sources.jsonl.gz` с атрибуцией. Лицензии диалогов, инструкций и задач записаны в их манифесте.
## Продолжение S AI LLM 1.0 с шага 10 000

Профиль [step10000_recovery.json](step10000_recovery.json) рассчитан на CopyLLM 1.276B с исходным словарём на 65 536 токенов. Первый запуск загружает все веса из `step-010000.pt` и создаёт новый AdamW для нового расписания обучения. Исходный файл хранится отдельно. Последующие запуски восстанавливают модель, оптимизатор, генераторы случайных чисел и позицию в расписании из `training-state.pt` той же стадии.

На каждом шаге модель обрабатывает 2 048 целевых токенов, по 1 024 на язык. Для RU и EN отдельно перемешиваются непересекающиеся окна по 256 целевых токенов. Окна выбираются по всей длине каждого потока и используются не более одного раза за стадию. Хвост короче 256 токенов отбрасывается. Профиль рассчитан на 1 024 новых шага, всего 2 097 152 целевых токена. Позиция в расписании определяется сохранённым шагом. Повторное использование данных из первых 10 000 шагов возможно.

### Запуск после согласования GPU-бюджета

На Linux разместите код отдельно от данных. Выполняйте команды из корня репозитория. Профиль использует одну GPU с нативным BF16 и полный AdamW. Для этой конфигурации используйте выделенную GPU класса 96 ГБ и оставьте не менее 40 ГБ на локальном диске для двух поколений состояния и временных файлов.

```bash
PY=/root/S-AI-LLM/.venv/bin/python
DATA=/root/S-AI-Training/sai-1b-vocab65536
BASE="$DATA/long-run/checkpoints/step-010000.pt"
RUN="$DATA/recovery-step10000"
PROFILE=training_1b/step10000_recovery.json

# Пример утверждённого лимита: до 128 новых шагов и 15 минут на процесс.
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900s "$PY" -m training_1b.pretrain \
  --profile "$PROFILE" --corpus "$DATA/packed-v2" --output "$RUN" \
  --init "$BASE" --stop-after 128 --max-session-tokens 262144 --max-session-seconds 780
```

При достижении мягкого лимита обучение останавливается между шагами и сохраняет последний завершённый шаг. Время учитывается с открытия корпуса и включает загрузку, обучение, штатную валидацию и сохранение. В примере разница между лимитами оставляет 120 секунд на завершение работы. Жёсткий предел задаёт [GNU timeout](https://www.gnu.org/software/coreutils/manual/html_node/timeout-invocation.html). При принудительном завершении действителен последний успешно заменённый `training-state.pt`. Состояние сохраняется каждые 32 шага и при плановой остановке. При остановке до первого сохранения повторите команду с `--init`.

Для возобновления возьмите номер стадии из последней строки `"event": "saved"`. В примере ниже она дошла до 128. Если номер другой, укажите фактическое значение. `--stop-after` задаёт итоговый шаг стадии. Токенный лимит считается для дополнительных шагов этой сессии.

```bash
SAVED=128
NEXT=$((SAVED + 128))
# NEXT должен быть не больше 1024.
CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 900s "$PY" -m training_1b.pretrain \
  --profile "$PROFILE" --corpus "$DATA/packed-v2" --output "$RUN" \
  --resume "$RUN/training-state.pt" --stop-after "$NEXT" \
  --max-session-tokens 262144 --max-session-seconds 780
```

Каждая сессия расходует отдельный лимит времени. Стоимость включает время аренды GPU до и после работы процесса. Жёсткий лимит завершает процесс, а аренда управляется отдельно у провайдера.

### Генерация

Для исходного чекпойнта и новых состояний стадии генератор использует словарный выход CopyLLM, соответствующий цели обучения. По умолчанию он снижает вероятность недавних токенов продолжения и блокирует повтор трёхтокенных последовательностей до выбора top-k. EOS завершает генерацию. Для других форматов остаются их прежние настройки. Эти меры уменьшают циклы повторов. Связность текста зависит от весов модели.

```bash
"$PY" generate.py --checkpoint "$BASE" --device cuda \
  --prompt "Сегодня утром я проснулся и" --max-new-tokens 48 \
  --top-k 40 --temperature 0.85 --repetition-penalty 1.2 --no-repeat-ngram-size 3
```

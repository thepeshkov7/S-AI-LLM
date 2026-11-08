# Предобучение S AI LLM 1.0

[Отчёт и результаты](REPORT.md). Исходный `CopyLLM` содержит 1 033 003 011 параметров. Файлы весов загружаются с `weights_only=True`.

## Colab

Откройте [ноутбук T4](S_AI_LLM_1_0_Pretraining_T4.ipynb), выберите Tesla T4 и выполняйте ячейки по порядку. В режиме отдельных файлов загрузите в `/content` исходный `s_ai_llm_1_0_untrained_expanded.pt` и, при наличии, `sai1b_wikipedia_pilot_10m.tar.gz`. Папка результатов — `/content/sai1b-results`. Скачайте состояние и метрики до завершения сессии. Для работы через Google Drive включите `USE_DRIVE` в первой кодовой ячейке. Тогда результаты будут в `MyDrive/S_AI_Training/S_AI_LLM_1_0/pretraining_v3_20261108`.

Пилот сохраняет состояние после шага 12 и продолжает до шага 24. Для возобновления в новом сеансе выполните ячейки настройки и корпуса, затем ячейку возобновления. В режиме отдельных файлов сначала верните сохранённый `training-state.pt` в `/content/sai1b-results/t4-pilot`.

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

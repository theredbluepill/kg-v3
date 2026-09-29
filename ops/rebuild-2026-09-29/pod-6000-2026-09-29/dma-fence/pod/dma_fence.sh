#!/bin/bash
# Task A: DMA fence test + source mutation check. No set -e: every step is recorded.
R=/root/receipts/dma-fence
mkdir -p $R
exec > >(tee $R/driver.log) 2>&1
export CUDA_VISIBLE_DEVICES=0
PY=/root/kg-v3/.venv/bin/python
T=tests/kaggriculture/test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma
date -u +"start %FT%TZ"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
cd /root/kg-v3 && git rev-parse HEAD > $R/git_state.txt && git status --porcelain | wc -l >> $R/git_state.txt
sha256sum python/owl/kaggriculture/env.py tests/kaggriculture/test_env_cuda_fence.py tests/kaggriculture/test_env.py python/owl/rs.abi3.so > $R/hashes_original.sha256
for i in 1 2 3; do
  date -u +"step1 run $i %FT%TZ"
  timeout 600 $PY -m pytest -p no:cacheprovider "$T" -v -rA > $R/step1_original_run$i.log 2>&1
  echo "step1 run $i exit=$?"
done
# Step 2: mutation in a scratch copy
rm -rf /root/fence-scratch
mkdir -p /root/fence-scratch
tar -C /root/kg-v3 --exclude=./.venv --exclude=./target --exclude=./.git -cf - . | tar -C /root/fence-scratch -xf -
cp /root/fence-scratch/python/owl/kaggriculture/env.py /root/env.py.orig
cd /root/fence-scratch
grep -c "            torch.cuda.current_stream(self.transfer_device).synchronize()" python/owl/kaggriculture/env.py > $R/step2_match_count.txt
sed -i 's/^            torch.cuda.current_stream(self.transfer_device).synchronize()$/            pass  # MUTATION: fence removed/' python/owl/kaggriculture/env.py
diff /root/env.py.orig python/owl/kaggriculture/env.py > $R/step2_mutation.diff
sha256sum python/owl/kaggriculture/env.py > $R/step2_mutated.sha256
PYTHONPATH=/root/fence-scratch/python $PY -c "import owl, owl.kaggriculture.env as e, owl.rs; print(owl.__file__); print(e.__file__); print(owl.rs.__file__)" > $R/step2_import_paths.txt 2>&1
date -u +"step2 mutated %FT%TZ"
PYTHONPATH=/root/fence-scratch/python timeout 600 $PY -m pytest -p no:cacheprovider "$T" -v -rA > $R/step2_mutated.log 2>&1
echo "step2 mutated exit=$?"
# Step 3: restore byte-for-byte
cp /root/env.py.orig python/owl/kaggriculture/env.py
sha256sum python/owl/kaggriculture/env.py /root/kg-v3/python/owl/kaggriculture/env.py > $R/step3_restored.sha256
cmp python/owl/kaggriculture/env.py /root/kg-v3/python/owl/kaggriculture/env.py && echo "step3 cmp identical"
date -u +"step3 restored %FT%TZ"
PYTHONPATH=/root/fence-scratch/python timeout 600 $PY -m pytest -p no:cacheprovider "$T" -v -rA > $R/step3_restored.log 2>&1
echo "step3 restored exit=$?"
cd /root && rm -rf /root/fence-scratch /root/env.py.orig
cd /root/kg-v3 && git status --porcelain | wc -l > $R/git_state_after.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
date -u +"end %FT%TZ"
echo DONE

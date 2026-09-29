#!/bin/bash
# Repeat of step 2/3 with no copied bytecode, so tracebacks name the scratch files.
R=/root/receipts/dma-fence
exec > >(tee $R/driver_2b.log) 2>&1
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
PY=/root/kg-v3/.venv/bin/python
T=tests/kaggriculture/test_env_cuda_fence.py::test_step_does_not_overwrite_pending_dma
date -u +"start2b %FT%TZ"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
rm -rf /root/fence-scratch; mkdir -p /root/fence-scratch
tar -C /root/kg-v3 --exclude=./.venv --exclude=./target --exclude=./.git --exclude=__pycache__ -cf - . | tar -C /root/fence-scratch -xf -
echo "pycache dirs in scratch: $(find /root/fence-scratch -name __pycache__ | wc -l)"
cp /root/fence-scratch/python/owl/kaggriculture/env.py /root/env.py.orig
cd /root/fence-scratch
sed -i 's/^            torch.cuda.current_stream(self.transfer_device).synchronize()$/            pass  # MUTATION: fence removed/' python/owl/kaggriculture/env.py
diff /root/env.py.orig python/owl/kaggriculture/env.py > $R/step2b_mutation.diff
PYTHONPATH=/root/fence-scratch/python timeout 600 $PY -m pytest -p no:cacheprovider "$T" -v -rA > $R/step2b_mutated.log 2>&1
echo "step2b mutated exit=$?"
cp /root/env.py.orig python/owl/kaggriculture/env.py
sha256sum python/owl/kaggriculture/env.py /root/kg-v3/python/owl/kaggriculture/env.py > $R/step3b_restored.sha256
cmp python/owl/kaggriculture/env.py /root/kg-v3/python/owl/kaggriculture/env.py && echo "step3b cmp identical"
PYTHONPATH=/root/fence-scratch/python timeout 600 $PY -m pytest -p no:cacheprovider "$T" -v -rA > $R/step3b_restored.log 2>&1
echo "step3b restored exit=$?"
echo "pycache dirs in scratch after: $(find /root/fence-scratch -name __pycache__ | wc -l)"
cd /root && rm -rf /root/fence-scratch /root/env.py.orig
(cd /root/kg-v3 && git status --porcelain | wc -l)
date -u +"end2b %FT%TZ"
echo DONE

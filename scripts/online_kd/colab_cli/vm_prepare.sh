#!/bin/bash
# Runs ON the Colab VM (uploaded to /content/okd/ with online_kd.py, apply_online_patch.py, score_fold0.py, vm_check.py).
# Downloads the kit, the Raptor teacher, the competition train.csv and the d96 cache; runs the teammate's pinned setup;
# installs the online-distillation patch and the lane env files. Idempotent: finished parts are skipped.
# Needs /content/kaggle.env (KAGGLE_API_TOKEN=...). Status: /content/okd/prepare.status (RUNNING / DONE / FAILED <step>).
set -uo pipefail
S=/content/okd/prepare.status; L=kd_e4on
fail() { echo "FAILED $1" > $S; exit 1; }
echo RUNNING > $S
export KAGGLE_API_TOKEN=$(cut -d= -f2- /content/kaggle.env)
pip install -q kaggle==2.2.2 || fail pip
# Kaggle's whole-dataset download returns 404 for some shared private datasets; fall back to per-file downloads.
dl_dataset() {   # dl_dataset <owner/slug> <dest>
  kaggle datasets download "$1" -p "$2" --unzip -q && return 0
  kaggle datasets files "$1" --page-size 1000 --csv | tail -n +2 | cut -d, -f1 > /content/okd/files_$(basename $1).txt || return 1
  [ -s /content/okd/files_$(basename $1).txt ] || return 1
  export DS="$1" DEST="$2"
  xargs -P 8 -I{} sh -c 'd="$DEST/$(dirname "{}")"; mkdir -p "$d"; for t in 1 2 3; do kaggle datasets download "$DS" -f "{}" -p "$d" -q >/dev/null 2>&1 && break; done; z="$d/$(basename "{}").zip"; if [ -f "$z" ]; then unzip -oq "$z" -d "$d" && rm -f "$z"; fi; [ -f "$d/$(basename "{}")" ] || echo "MISSING {}"' \
    < /content/okd/files_$(basename $1).txt | tee /content/okd/missing_$(basename $1).txt
  [ ! -s /content/okd/missing_$(basename $1).txt ]
}
cd /content
# the kit: per-file list, then a sequential rate-limit-aware fetch of anything missing (parallel fetches hit HTTP 429)
[ -s /content/okd/files_rsna-pubkd-teammate-kit.txt ] || kaggle datasets files tumis001/rsna-pubkd-teammate-kit --page-size 1000 --csv \
  | tail -n +2 | cut -d, -f1 > /content/okd/files_rsna-pubkd-teammate-kit.txt || fail kitlist
python3 /content/okd/kit_seq.py; tail -1 /content/okd/kit_seq.log | grep -q "KIT_DONE missing 0" || fail kit
[ -f raptor/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt ] || \
  dl_dataset nartaa/rsna-knee-publication-swa-weights-20261007 raptor || fail raptor
if [ ! -f comp/train.csv ]; then
  kaggle competitions download rsna-knee-abnormality-detection -f train.csv -p comp -q || fail traincsv
  (cd comp && { [ -f train.csv.zip ] && unzip -oq train.csv.zip && rm -f train.csv.zip; true; })
fi
mkdir -p d96
for i in 0 1 2 3; do
  if [ ! -f d96/shard_$i/cache_summary.json ]; then
    ok=0
    for try in 1 2 3; do
      rm -rf _dl_$i && mkdir _dl_$i
      kaggle kernels output tumis001/rsna-coatnet-d96-cache-shard-$i -p _dl_$i > /dev/null && { ok=1; break; }
    done
    [ $ok = 1 ] || fail "cache shard $i"
    rm -rf d96/shard_$i && mv _dl_$i/d96 d96/shard_$i && rm -rf _dl_$i
  fi
  echo "cache shard $i ok"
done
[ -f bundle.tgz ] || tar czf bundle.tgz -C kit work || fail bundle
python3 kit/work/scripts/colab_pubkd_setup.py > /content/okd/setup.log 2>&1 || fail setup
grep -q SETUP_OK /content/okd/setup.log || fail setup_ok
# online-distillation code
cp /content/okd/online_kd.py /content/okd/apply_online_patch.py /content/okd/score_fold0.py /work/scripts/
grep -q V4_ONLINE /work/scripts/r50_gl4_runner.py || python3 /work/scripts/apply_online_patch.py /work/scripts/r50_gl4_runner.py || fail patch
mkdir -p /work/online/clean
cp raptor/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt raptor/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt /work/online/
cp kit/scoring/rsna_root/repository/data/coatnet_d96_fold0/pubkd/raptor384_*.json \
   kit/scoring/rsna_root/repository/data/coatnet_d96_fold0/pubkd/raptor224_*.json /work/online/clean/
ONLINE="V4_ONLINE=1
V4_ONLINE_CKPT384=/work/online/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
V4_ONLINE_CKPT224=/work/online/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
V4_ONLINE_T224=1
V4_ONLINE_CLEAN=/work/online/clean
V4_ONLINE_PURE=/work/pubkd/pubkd_targets_fullorder_rapvalues.json
V4_ONLINE_MIX=0.3
V4_ONLINE_CLIP=4.0
V4_ONLINE_ALPHA=1.0"
grep -v '^GL_OUT=' /work/env/pubkd_kd_e4.env > /work/env/pubkd_$L.env
printf '%s\nGL_OUT=/work/out/pubkd_%s\n' "$ONLINE" "$L" >> /work/env/pubkd_$L.env
grep -v '^GL_OUT=' /work/env/pubkd_smoke_kd_e4.env > /work/env/pubkd_smoke_$L.env
printf '%s\nGL_OUT=/work/out/pubkd_smoke_%s\n' "$ONLINE" "$L" >> /work/env/pubkd_smoke_$L.env
echo DONE > $S

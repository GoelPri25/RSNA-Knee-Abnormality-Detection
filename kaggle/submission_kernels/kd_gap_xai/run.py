"""KD-GAP xAI (no training, no submission). Built by scripts/build_kaggle_kd_gap_kernel.py; kd_sub = the E4 submission
script (kaggle/submission_kernels/pubkd_kd_e4_submission/submission.py): DICOM builder, windows, r50 l4copy model."""
import base64
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EMB = {"kd_sub.py": "IiIicjUwLUdMIHY0IGw0Y29weSBzMjAyNiBzaW5nbGUtbW9kZWwgc3VibWlzc2lvbiAobGVkZ2VyIHJvd3MgcjUwLUdMNCwgU0wtMHY0KS4gRm9sZCAwIChsYWJlbC12MiBjb25maWRlbnQgY2VsbHMpOgpmdXNlZCAwLjk0NzUgdnMgbGFiZWwtdjIgcjUwIDAuOTQyNSAoTEIgMC45MzcpLgoKT25lIFJlc05ldC01MCByZWFkcyBlYWNoIHN0dWR5IHR3aWNlOgogIGdsb2JhbCAgdGhlIDk0IGV2YWwgd2luZG93cyBvdmVyIHRoZSB2YWxpZCByYW5nZSwgMzIwIGNlbnRyZSBjcm9wIChsYWJlbC12MiAvIEUzIHBhdGgpIC0+IHdpbmRvdyBhdHRlbnRpb24gaGVhZCBnOwogIGxvY2FsICAgZm9yIGVhY2ggb2YgdGhlIDEwIGNyb3BwZWQgZmluZGluZ3MsIDQgem9vbWVkIGNyb3BzICgxOTIgcHggYm94LCAxMjggcHggZm9yIG1lZGlhbCBtZW5pc2N1cyAvIGNvbnR1c2lvbiwKICAgICAgICAgIHJlc2FtcGxlZCB0byAzODQpIHJ1biB0aHJvdWdoIHRoZSBzaGFyZWQgc3RlbS1sYXllcjMgKGNyb3AgQk4gc3RhdGlzdGljcykgYW5kIHRoZSBjcm9wcycgb3duIGxheWVyNCAtPgogICAgICAgICAgbG9jYWwgaGVhZCBsLCB3ZWlnaHRlZCBieSB0aGUgd2luZG93IGF0dGVudGlvbiB3aXRoaW4gdGhlIGZpbmRpbmcuCiAgZnVzZWQgICBzaWdtb2lkKChnICsgbCkgLyAyKSBvbiB0aGUgMTAgY3JvcHBlZCBmaW5kaW5ncywgc2lnbW9pZChnKSBvbiBlZmZ1c2lvbiAvIHN5bm92aXRpcy4KQ3JvcCBwbGFjZW1lbnQgaXMgc2VsZi1sb2NhdGVkIChTTC0wdjQpLCBubyBNUzE6IHBlciBmaW5kaW5nIHRoZSA0IHdpbmRvd3Mgd2l0aCB0aGUgbGFyZ2VzdCBnbG9iYWwgYXR0ZW50aW9uLCBvbiBlYWNoIHRoZQpncmFkaWVudCB4IGFjdGl2YXRpb24gY2VudHJvaWQgb2YgdGhhdCB3aW5kb3cncyBsb2dpdCBjb250cmlidXRpb24gb3ZlciB0aGUgMTAgeCAxMCBsYXllci00IGNlbGxzLgpWb2x1bWU6IEUzIGtlcm5lbCdzIGQ5NiBidWlsZGVyLCB1bmNoYW5nZWQuIGZwMTYgYXV0b2Nhc3Qgd2l0aCBhbiBmcDMyIHJldHJ5IGZvciBub24tZmluaXRlIGxvZ2l0czsgYSBzdHVkeSB0aGF0CmNhbm5vdCBiZSByZWFkIGtlZXBzIHRoZSBzYW1wbGVfc3VibWlzc2lvbiB2YWx1ZXMuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgY29weQppbXBvcnQgY3N2CmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCBzeXMKaW1wb3J0IHRpbWUKZnJvbSBjb25jdXJyZW50LmZ1dHVyZXMgaW1wb3J0IFRocmVhZFBvb2xFeGVjdXRvcgpmcm9tIG11bHRpcHJvY2Vzc2luZyBpbXBvcnQgUG9vbApmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgdG9yY2gKaW1wb3J0IHRvcmNoLm5uIGFzIG5uCmltcG9ydCB0b3JjaC5ubi5mdW5jdGlvbmFsIGFzIEYKClRBUkdFVFMgPSBbIkFDTCIsICJNQ0wiLCAiTWVkaWFsIE1lbmlzY3VzIiwgIkxhdGVyYWwgTWVuaXNjdXMiLCAiTWVkaWFsIE9BIiwgIkxhdGVyYWwgT0EiLAogICAgICAgICAgICJQRiBPQSIsICJFZmZ1c2lvbiIsICJTeW5vdml0aXMiLCAiQmFrZXIncyIsICJDb250dXNpb24iLCAiRnJhY3R1cmUiXQpJTUcsIENST1BfTU0sIEVWQUxfUkVTLCBFVkFMX1dJTkRPV1MgPSAzODQsIDE0MC4wLCAzMjAsIDk0ClNMT1RTID0gWygiU2FnaXR0YWwiLCAxLCAyNiksICgiU2FnaXR0YWwiLCAwLCAyMiksICgiQ29yb25hbCIsIDEsIDE4KSwgKCJDb3JvbmFsIiwgMCwgMTIpLCAoIkF4aWFsIiwgLTEsIDE4KV0KTUFYUyA9IHN1bShrIGZvciBfLCBfLCBrIGluIFNMT1RTKQpCT1ggPSB7IkFDTCI6IDE5MiwgIk1DTCI6IDE5MiwgIk1lZGlhbCBNZW5pc2N1cyI6IDEyOCwgIkxhdGVyYWwgTWVuaXNjdXMiOiAxOTIsICJNZWRpYWwgT0EiOiAxOTIsICJMYXRlcmFsIE9BIjogMTkyLAogICAgICAgIlBGIE9BIjogMTkyLCAiQmFrZXIncyI6IDE5MiwgIkNvbnR1c2lvbiI6IDEyOCwgIkZyYWN0dXJlIjogMTkyfQpDRiA9IG5wLmFycmF5KFtrIGZvciBrLCB0IGluIGVudW1lcmF0ZShUQVJHRVRTKSBpZiB0IGluIEJPWF0pCkJPWE4gPSBucC5hcnJheShbQk9YLmdldCh0LCAwKSBmb3IgdCBpbiBUQVJHRVRTXSkKQ01BU0sgPSBucC5pc2luKG5wLmFyYW5nZSgxMiksIENGKQpUT1AsIENST1BfUkVTID0gNCwgMzg0CldFSUdIVFMgPSB7ImRhdGFzZXQiOiAidHVtaXMwMDEvcnNuYS1wdWJrZC1rZC1lNC1mb2xkMCIsICJmaWxlIjogIndlaWdodHNfc3dhMjFfMjUucHQiLAogICAgICAgICAgICJzaGEyNTYiOiAiMDgwY2Y2M2Y4OTk3YTc2OGQwYjYyOTBlNzFhNTk3YzUzZDdiZTAyYzMwNTY0ZGY0N2VlZTk1MWMyMDM5OGEzNiJ9Ck1FQU4gPSB0b3JjaC50ZW5zb3IoWzAuNDg1LCAwLjQ1NiwgMC40MDZdKS52aWV3KDEsIDMsIDEsIDEpClNURCA9IHRvcmNoLnRlbnNvcihbMC4yMjksIDAuMjI0LCAwLjIyNV0pLnZpZXcoMSwgMywgMSwgMSkKCgpkZWYgY29tcGV0aXRpb25fZGlyKCk6CiAgICByZXR1cm4gbmV4dChwIGZvciBwIGluIChQYXRoKCIva2FnZ2xlL2lucHV0L2NvbXBldGl0aW9ucy9yc25hLWtuZWUtYWJub3JtYWxpdHktZGV0ZWN0aW9uIiksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBQYXRoKCIva2FnZ2xlL2lucHV0L3JzbmEta25lZS1hYm5vcm1hbGl0eS1kZXRlY3Rpb24iKSkgaWYgcC5leGlzdHMoKSkKCgpkZWYgZGF0YXNldF9kaXIocmVmKToKICAgIG93bmVyLCBzbHVnID0gcmVmLnNwbGl0KCIvIikKICAgIHJldHVybiBuZXh0KHAgZm9yIHAgaW4gKFBhdGgoIi9rYWdnbGUvaW5wdXQvZGF0YXNldHMiKSAvIG93bmVyIC8gc2x1ZywgUGF0aCgiL2thZ2dsZS9pbnB1dCIpIC8gc2x1ZykgaWYgcC5leGlzdHMoKSkKCgpkZWYgcmVhZF9jc3YocGF0aCk6CiAgICB3aXRoIHBhdGgub3BlbihuZXdsaW5lPSIiLCBlbmNvZGluZz0idXRmLTgtc2lnIikgYXMgc3RyZWFtOgogICAgICAgIHJldHVybiBsaXN0KGNzdi5EaWN0UmVhZGVyKHN0cmVhbSkpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIGQ5NiBidWlsZGVyIChFMyBrZXJuZWwsIHVuY2hhbmdlZCkKZGVmIGhlYWRlcihwYXRoKToKICAgIGltcG9ydCBweWRpY29tCiAgICB0cnk6CiAgICAgICAgaCA9IHB5ZGljb20uZGNtcmVhZChwYXRoLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSkKICAgICAgICBpb3AgPSBnZXRhdHRyKGgsICJJbWFnZU9yaWVudGF0aW9uUGF0aWVudCIsIE5vbmUpCiAgICAgICAgaXBwID0gZ2V0YXR0cihoLCAiSW1hZ2VQb3NpdGlvblBhdGllbnQiLCBOb25lKQogICAgICAgIGlmIGlvcCBpcyBub3QgTm9uZSBhbmQgaXBwIGlzIG5vdCBOb25lIGFuZCBsZW4oaW9wKSA9PSA2OgogICAgICAgICAgICByLCBjID0gbnAuYXJyYXkoaW9wWzozXSwgZmxvYXQpLCBucC5hcnJheShpb3BbMzpdLCBmbG9hdCkKICAgICAgICAgICAgcG9zID0gZmxvYXQobnAuZG90KG5wLmFycmF5KGlwcCwgZmxvYXQpLCBucC5jcm9zcyhyLCBjKSkpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcG9zID0gZmxvYXQoZ2V0YXR0cihoLCAiSW5zdGFuY2VOdW1iZXIiLCAwKSBvciAwKQogICAgICAgIHBzID0gZ2V0YXR0cihoLCAiUGl4ZWxTcGFjaW5nIiwgTm9uZSkKICAgICAgICByZXR1cm4gcG9zLCBwYXRoLCAoZmxvYXQocHNbMF0pIGlmIHBzIGlzIG5vdCBOb25lIGVsc2UgMC41KSwgVHJ1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gMC4wLCBwYXRoLCAwLjUsIEZhbHNlCgoKZGVmIHBpeGVscyhwYXRoKToKICAgIGltcG9ydCBweWRpY29tCiAgICBmcm9tIHB5ZGljb20ucGl4ZWxzIGltcG9ydCBhcHBseV9tb2RhbGl0eV9sdXQKICAgIGQgPSBweWRpY29tLmRjbXJlYWQocGF0aCkKICAgIGEgPSBhcHBseV9tb2RhbGl0eV9sdXQoZC5waXhlbF9hcnJheSwgZCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICBpZiBzdHIoZ2V0YXR0cihkLCAiUGhvdG9tZXRyaWNJbnRlcnByZXRhdGlvbiIsICIiKSkgPT0gIk1PTk9DSFJPTUUxIjoKICAgICAgICBhID0gYS5tYXgoKSAtIGEKICAgIHJldHVybiBhCgoKZGVmIGNyb3BfcmVzaXplKGEsIHBzKToKICAgIGltcG9ydCBjdjIKICAgIGgsIHcgPSBhLnNoYXBlCiAgICBjcHggPSBtaW4oaW50KHJvdW5kKENST1BfTU0gLyBtYXgocHMsIDFlLTMpKSksIGgsIHcpCiAgICB5MCwgeDAgPSAoaCAtIGNweCkgLy8gMiwgKHcgLSBjcHgpIC8vIDIKICAgIHJldHVybiBjdjIucmVzaXplKGFbeTA6eTAgKyBjcHgsIHgwOngwICsgY3B4XSwgKElNRywgSU1HKSwgaW50ZXJwb2xhdGlvbj1jdjIuSU5URVJfQVJFQSkKCgpkZWYgYnVpbGRfc3R1ZHkoam9iKToKICAgIHN0dWR5X2lkLCByb3dzLCBzZXJpZXNfZGlyID0gam9iICAgICAgICAgICAgICAgICAgICAgIyBzZXJpZXNfZGlyOiA8Y29tcGV0aXRpb24+L3Rlc3Rfc2VyaWVzICh0cmFpbiBmb3IgY2hlY2tzKQogICAgdm9sID0gbnAuemVyb3MoKE1BWFMsIElNRywgSU1HKSwgbnAudWludDgpCiAgICBpZHgsIHVzZWQsIGVycm9ycyA9IDAsIHNldCgpLCAwCiAgICB3aXRoIFRocmVhZFBvb2xFeGVjdXRvcihtYXhfd29ya2Vycz04KSBhcyBwb29sOgogICAgICAgIGZvciBwbGFuZSwgZmx1aWQsIGsgaW4gU0xPVFM6CiAgICAgICAgICAgIGNhbmRzID0gW3IgZm9yIHIgaW4gcm93cyBpZiByWyJBbmF0b21pY2FsX1BsYW5lIl0gPT0gcGxhbmUgYW5kIHJbIlNlcmllc0luc3RhbmNlVUlEIl0gbm90IGluIHVzZWRdCiAgICAgICAgICAgIHByZWYgPSBbciBmb3IgciBpbiBjYW5kcyBpZiBmbHVpZCBpbiAoMCwgMSkgYW5kIGludChyLmdldCgiRmx1aWRfU2Vuc2l0aXZlIiwgMCkgb3IgMCkgPT0gZmx1aWRdCiAgICAgICAgICAgIHJvdyA9IChwcmVmIG9yIGNhbmRzIG9yIFtOb25lXSlbMF0KICAgICAgICAgICAgaWYgcm93IGlzIE5vbmU6CiAgICAgICAgICAgICAgICBpZHggKz0gawogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgdXNlZC5hZGQocm93WyJTZXJpZXNJbnN0YW5jZVVJRCJdKQogICAgICAgICAgICBmaWxlcyA9IHNvcnRlZCgoc2VyaWVzX2RpciAvIHN0dWR5X2lkIC8gcm93WyJTZXJpZXNJbnN0YW5jZVVJRCJdKS5nbG9iKCIqLmRjbSIpKQogICAgICAgICAgICBoZWFkcyA9IGxpc3QocG9vbC5tYXAoaGVhZGVyLCBmaWxlcykpCiAgICAgICAgICAgIG9rX3BzID0gW2hbMl0gZm9yIGggaW4gaGVhZHMgaWYgaFszXV0KICAgICAgICAgICAgbWVkX3BzID0gZmxvYXQobnAubWVkaWFuKG9rX3BzKSkgaWYgb2tfcHMgZWxzZSAwLjUKICAgICAgICAgICAgaGVhZHMuc29ydChrZXk9bGFtYmRhIGg6IGhbMF0pCiAgICAgICAgICAgIG4gPSBsZW4oaGVhZHMpCiAgICAgICAgICAgIGlmIG4gPT0gMDoKICAgICAgICAgICAgICAgIGlkeCArPSBrCiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBsbywgaGkgPSBpbnQobiAqIDAuMDIpLCBpbnQobiAqIDAuOTgpIC0gMQogICAgICAgICAgICBoaSA9IG1heChoaSwgbG8pCiAgICAgICAgICAgIHBpY2tzID0gW2ludChtaW4ocCwgbiAtIDEpKSBmb3IgcCBpbiAobnAubGluc3BhY2UobG8sIGhpLCBrKS5yb3VuZCgpLmFzdHlwZShpbnQpIGlmIG4gPiAxIGVsc2UgWzBdICogayldCgogICAgICAgICAgICBkZWYgbG9hZChwKToKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICByZXR1cm4gcGl4ZWxzKGhlYWRzW3BdWzFdKSwgaGVhZHNbcF1bMl0KICAgICAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIE5vbmUsIG1lZF9wcwogICAgICAgICAgICBnb3QgPSBsaXN0KHBvb2wubWFwKGxvYWQsIHBpY2tzKSkKICAgICAgICAgICAgdmFsaWQgPSBbYSBmb3IgYSwgXyBpbiBnb3QgaWYgYSBpcyBub3QgTm9uZV0KICAgICAgICAgICAgZXJyb3JzICs9IGxlbihnb3QpIC0gbGVuKHZhbGlkKQogICAgICAgICAgICBsb3EsIGhpcSA9IChucC5wZXJjZW50aWxlKG5wLmNvbmNhdGVuYXRlKFthLnJhdmVsKCkgZm9yIGEgaW4gdmFsaWRdKSwgWzIuMCwgOTguMF0pIGlmIHZhbGlkIGVsc2UgKDAuMCwgMS4wKSkKICAgICAgICAgICAgZm9yIGEsIHBzIGluIGdvdDoKICAgICAgICAgICAgICAgIGlmIGEgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICAgICAgYXcgPSBucC5jbGlwKChhIC0gbG9xKSAvIChoaXEgLSBsb3EgKyAxZS02KSwgMCwgMSkKICAgICAgICAgICAgICAgICAgICB2b2xbaWR4XSA9IChjcm9wX3Jlc2l6ZShhdywgcHMgaWYgcHMgPiAwIGVsc2UgbWVkX3BzKSAqIDI1NSkuYXN0eXBlKG5wLnVpbnQ4KQogICAgICAgICAgICAgICAgaWR4ICs9IDEKICAgIG1hc2sgPSB2b2wucmVzaGFwZShNQVhTLCAtMSkubWF4KDEpID4gMAogICAgcmV0dXJuIHN0dWR5X2lkLCB2b2wsIG1hc2ssIGVycm9ycwoKCmRlZiB1bmlmb3JtX2NlbnRyZXMobWFzayk6CiAgICAiIiI5NCBjZW50cmVzIG92ZXIgdGhlIHdob2xlIHZhbGlkIHJhbmdlIChjb2F0bmV0X2Q5Nl9ydW5uZXIuY2VudHJlcywgZXZhbCksIGNsaXBwZWQgdG8gWzEsIE1BWFMgLSAyXS4iIiIKICAgIHZhbGlkID0gbnAud2hlcmUobWFzaylbMF0KICAgIGlmIGxlbih2YWxpZCkgPCAzOgogICAgICAgIHZhbGlkID0gbnAuYXJhbmdlKDMpCiAgICBsbywgaGkgPSBpbnQodmFsaWQubWluKCkpLCBpbnQodmFsaWQubWF4KCkpCiAgICBjcyA9IG5wLmFyYW5nZShsbyArIDEsIGhpKSBpZiBoaSAtIGxvID49IDIgZWxzZSBucC5hcnJheShbbWF4KDEsIG1pbigobG8gKyBoaSkgLy8gMiwgTUFYUyAtIDIpKV0pCiAgICByZXR1cm4gbnAuY2xpcChjc1tucC5saW5zcGFjZSgwLCBsZW4oY3MpIC0gMSwgRVZBTF9XSU5ET1dTKS5yb3VuZCgpLmFzdHlwZShpbnQpXSwgMSwgTUFYUyAtIDIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIG1vZGVsIChyNTBfZ2w0X3J1bm5lci5SYXB0b3JHTDQsIGFybSBsNGNvcHkpCmNsYXNzIER1YWxCTihubi5CYXRjaE5vcm0yZCk6CiAgICAiIiJCYXRjaE5vcm0yZCB3aXRoIGEgc2Vjb25kIHNldCBvZiBydW5uaW5nIHN0YXRpc3RpY3MgZm9yIGNyb3AgaW1hZ2VzIChjcm9wID0gVHJ1ZSk7IGFmZmluZSBzaGFyZWQuIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGJuKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKGJuLm51bV9mZWF0dXJlcywgYm4uZXBzLCBibi5tb21lbnR1bSwgYm4uYWZmaW5lLCBibi50cmFja19ydW5uaW5nX3N0YXRzKQogICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCJjcm9wX21lYW4iLCB0b3JjaC56ZXJvcyhibi5udW1fZmVhdHVyZXMpKQogICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCJjcm9wX3ZhciIsIHRvcmNoLm9uZXMoYm4ubnVtX2ZlYXR1cmVzKSkKICAgICAgICBzZWxmLnJlZ2lzdGVyX2J1ZmZlcigiY3JvcF9uYnQiLCB0b3JjaC56ZXJvcygoKSwgZHR5cGU9dG9yY2gubG9uZykpCiAgICAgICAgc2VsZi5jcm9wID0gRmFsc2UKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB4KToKICAgICAgICBpZiBub3Qgc2VsZi5jcm9wOgogICAgICAgICAgICByZXR1cm4gc3VwZXIoKS5mb3J3YXJkKHgpCiAgICAgICAgcmV0dXJuIEYuYmF0Y2hfbm9ybSh4LCBzZWxmLmNyb3BfbWVhbiwgc2VsZi5jcm9wX3Zhciwgc2VsZi53ZWlnaHQsIHNlbGYuYmlhcywgRmFsc2UsIDAuMCwgc2VsZi5lcHMpCgoKZGVmIHdyYXBfYm5zKG1vZHVsZSk6CiAgICBvdXQgPSBbXQogICAgZm9yIG5hbWUsIGNoIGluIG1vZHVsZS5uYW1lZF9jaGlsZHJlbigpOgogICAgICAgIGlmIGlzaW5zdGFuY2UoY2gsIG5uLkJhdGNoTm9ybTJkKToKICAgICAgICAgICAgZCA9IER1YWxCTihjaCk7IHNldGF0dHIobW9kdWxlLCBuYW1lLCBkKTsgb3V0LmFwcGVuZChkKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIG91dCArPSB3cmFwX2JucyhjaCkKICAgIHJldHVybiBvdXQKCgpjbGFzcyBWNEw0Q29weShubi5Nb2R1bGUpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIGltcG9ydCB0aW1tCiAgICAgICAgc2VsZi5iYWNrYm9uZSA9IHRpbW0uY3JlYXRlX21vZGVsKCJyZXNuZXQ1MC50dl9pbjFrIiwgcHJldHJhaW5lZD1GYWxzZSwgbnVtX2NsYXNzZXM9MCkKICAgICAgICBkID0gc2VsZi5iYWNrYm9uZS5udW1fZmVhdHVyZXMKICAgICAgICBzZWxmLm5vcm0gPSBubi5MYXllck5vcm0oZCkKICAgICAgICBzZWxmLmF0dCA9IG5uLlNlcXVlbnRpYWwobm4uTGluZWFyKGQsIDI1NiksIG5uLlRhbmgoKSwgbm4uRHJvcG91dCgwLjIpLCBubi5MaW5lYXIoMjU2LCAxMikpCiAgICAgICAgc2VsZi5jbHNXID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKDEyLCBkKSk7IHNlbGYuY2xzYiA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcygxMikpCiAgICAgICAgc2VsZi5ibnMgPSB3cmFwX2JucyhzZWxmLmJhY2tib25lKQogICAgICAgIGFzc2VydCBsZW4oc2VsZi5ibnMpID09IDUzCiAgICAgICAgc2VsZi5sYXllcjRfYyA9IGNvcHkuZGVlcGNvcHkoc2VsZi5iYWNrYm9uZS5sYXllcjQpCiAgICAgICAgZm9yIG0gaW4gc2VsZi5sYXllcjRfYy5tb2R1bGVzKCk6CiAgICAgICAgICAgIGlmIGlzaW5zdGFuY2UobSwgRHVhbEJOKToKICAgICAgICAgICAgICAgIG0uY3JvcCA9IFRydWUKICAgICAgICBzZWxmLm5vcm1fbCA9IG5uLkxheWVyTm9ybShkKQogICAgICAgIHNlbGYuY2xzViA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcygxMiwgZCkpOyBzZWxmLmNsc3ZiID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKDEyKSkKCiAgICBkZWYgY3JvcF9mZWF0dXJlcyhzZWxmLCB4KToKICAgICAgICBiYiA9IHNlbGYuYmFja2JvbmUKICAgICAgICBmb3IgbSBpbiBzZWxmLmJuczoKICAgICAgICAgICAgbS5jcm9wID0gVHJ1ZQogICAgICAgIHRyeToKICAgICAgICAgICAgaCA9IGJiLm1heHBvb2woYmIuYWN0MShiYi5ibjEoYmIuY29udjEoeCkpKSkKICAgICAgICAgICAgaCA9IHNlbGYubGF5ZXI0X2MoYmIubGF5ZXIzKGJiLmxheWVyMihiYi5sYXllcjEoaCkpKSkKICAgICAgICAgICAgcmV0dXJuIGJiLmZvcndhcmRfaGVhZChoKQogICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgIGZvciBtIGluIHNlbGYuYm5zOgogICAgICAgICAgICAgICAgbS5jcm9wID0gRmFsc2UKCgpkZWYgbG9hZF9tb2RlbChwYXRoLCBkZXZpY2UsIHNoYTI1Nj1Ob25lKToKICAgIGlmIHNoYTI1NiBpcyBub3QgTm9uZSBhbmQgaGFzaGxpYi5zaGEyNTYoUGF0aChwYXRoKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpICE9IHNoYTI1NjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJXZWlnaHRzIGhhc2ggbWlzbWF0Y2g6IHY0IGw0Y29weSIpCiAgICBtb2RlbCA9IFY0TDRDb3B5KCkKICAgIG1vZGVsLmxvYWRfc3RhdGVfZGljdCh0b3JjaC5sb2FkKHBhdGgsIG1hcF9sb2NhdGlvbj0iY3B1Iiwgd2VpZ2h0c19vbmx5PVRydWUpLCBzdHJpY3Q9VHJ1ZSkKICAgIG1vZGVsID0gbW9kZWwudG8oZGV2aWNlKS5ldmFsKCkudG8obWVtb3J5X2Zvcm1hdD10b3JjaC5jaGFubmVsc19sYXN0KQogICAgZm9yIHAgaW4gbW9kZWwucGFyYW1ldGVycygpOgogICAgICAgIHAucmVxdWlyZXNfZ3JhZF8oRmFsc2UpCiAgICByZXR1cm4gbW9kZWwKCgpkZWYgd2luZG93cyh2b2wsIGNzLCBkZXZpY2UpOgogICAgIiIiW2tdIGNlbnRyZSBzbGljZXMgLT4gdWludDggW2ssIDMsIElNRywgSU1HXSBvbiBkZXZpY2UgKHNsaWNlcyBjIC0gMSwgYywgYyArIDEpLiIiIgogICAgcmV0dXJuIHRvcmNoLmZyb21fbnVtcHkobnAuYXNjb250aWd1b3VzYXJyYXkodm9sW25wLnN0YWNrKFtjcyAtIDEsIGNzLCBjcyArIDFdLCAxKV0pKS50byhkZXZpY2UpCgoKZGVmIG5vcm1hbGlzZSh4KToKICAgIHJldHVybiAoKHggLSBNRUFOLnRvKHguZGV2aWNlKSkgLyBTVEQudG8oeC5kZXZpY2UpKS5jb250aWd1b3VzKG1lbW9yeV9mb3JtYXQ9dG9yY2guY2hhbm5lbHNfbGFzdCkKCgpkZWYgbG9jYXRlKG1vZGVsLCBmbSwgYSwgY3MpOgogICAgIiIic2VsZmxvY192NF9ncHUubG9jYXRlIG9uIHRoZSBnbG9iYWwgcGFzczogZm0gW0ssIDIwNDgsIGMsIGNdIGZwMzIsIGEgW0ssIDEyXSAtPiAoc2xpY2VzLCByb3dzLCBjb2xzLCB3ZWlnaHRzKQogICAgWzEwLCBUT1BdIGZvciB0aGUgY3JvcHBlZCBmaW5kaW5ncy4iIiIKICAgIGRldiA9IGZtLmRldmljZQogICAgVyA9IG1vZGVsLmNsc1cuZmxvYXQoKVtDRl0KICAgIGFjID0gYVs6LCBDRl0KICAgIHRvcCA9IGFjLnRvcGsoVE9QLCBkaW09MCkuaW5kaWNlcyAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIyBbVE9QLCAxMF0KICAgIFUgPSB0b3JjaC51bmlxdWUodG9wKTsgcm93ID0ge2ludChrKTogaiBmb3IgaiwgayBpbiBlbnVtZXJhdGUoVS50b2xpc3QoKSl9CiAgICB3aXRoIHRvcmNoLmVuYWJsZV9ncmFkKCk6CiAgICAgICAgdCA9IGZtW1VdLmRldGFjaCgpLmNsb25lKCkucmVxdWlyZXNfZ3JhZF8oVHJ1ZSkKICAgICAgICBjb250cmliID0gYWNbVV0gKiAobW9kZWwubm9ybS5mbG9hdCgpKHQubWVhbigoMiwgMykpKSBAIFcuVCkgICAgICAgICAgIyBbdSwgMTBdLCBhIGZpeGVkCiAgICAgICAgYyA9IHQuc2hhcGVbLTFdOyBweCA9IEVWQUxfUkVTIC8gYzsgZ2kgPSB0b3JjaC5hcmFuZ2UoYywgZGV2aWNlPWRldiwgZHR5cGU9dG9yY2guZmxvYXQzMikKICAgICAgICBsbyA9IChJTUcgLSBFVkFMX1JFUykgLy8gMgogICAgICAgIFMgPSBucC56ZXJvcygobGVuKENGKSwgVE9QKSwgbnAuaW50NjQpOyBSID0gUy5jb3B5KCk7IFEgPSBTLmNvcHkoKTsgQSA9IG5wLnplcm9zKChsZW4oQ0YpLCBUT1ApLCBucC5mbG9hdDMyKQogICAgICAgIGZvciBqIGluIHJhbmdlKGxlbihDRikpOgogICAgICAgICAgICBnciwgPSB0b3JjaC5hdXRvZ3JhZC5ncmFkKGNvbnRyaWJbOiwgal0uc3VtKCksIHQsIHJldGFpbl9ncmFwaD1qIDwgbGVuKENGKSAtIDEpCiAgICAgICAgICAgIGtzID0gdG9wWzosIGpdLnRvbGlzdCgpOyByb3dzID0gW3Jvd1trXSBmb3IgayBpbiBrc10KICAgICAgICAgICAgbXAgPSAoZ3Jbcm93c10gKiB0LmRldGFjaCgpW3Jvd3NdKS5zdW0oMSkuY2xhbXBfbWluKDApCiAgICAgICAgICAgIHRvdCA9IG1wLnN1bSgoMSwgMikpOyBvayA9IHRvdCA+IDA7IHRvdCA9IHRvdC5jbGFtcF9taW4oMWUtMTIpCiAgICAgICAgICAgIGN5ID0gdG9yY2gud2hlcmUob2ssIChtcC5zdW0oMikgKiBnaSkuc3VtKDEpIC8gdG90LCB0b3JjaC5mdWxsX2xpa2UodG90LCAoYyAtIDEpIC8gMikpCiAgICAgICAgICAgIGN4ID0gdG9yY2gud2hlcmUob2ssIChtcC5zdW0oMSkgKiBnaSkuc3VtKDEpIC8gdG90LCB0b3JjaC5mdWxsX2xpa2UodG90LCAoYyAtIDEpIC8gMikpCiAgICAgICAgICAgIFNbal0gPSBjc1trc10KICAgICAgICAgICAgUltqXSA9IG5wLnJvdW5kKChsbyArIChjeSArIDAuNSkgKiBweCkuY3B1KCkubnVtcHkoKSkuYXN0eXBlKG5wLmludDY0KQogICAgICAgICAgICBRW2pdID0gbnAucm91bmQoKGxvICsgKGN4ICsgMC41KSAqIHB4KS5jcHUoKS5udW1weSgpKS5hc3R5cGUobnAuaW50NjQpCiAgICAgICAgICAgIEFbal0gPSBhY1t0b3BbOiwgal0sIGpdLmRldGFjaCgpLmNwdSgpLm51bXB5KCkKICAgIHJldHVybiBTLCBSLCBRLCBBCgoKZGVmIHByZWRpY3Rfc3R1ZHkobW9kZWwsIHZvbCwgbWFzaywgZGV2aWNlLCBhbXA9VHJ1ZSwgdGFibGU9Tm9uZSk6CiAgICAiIiItPiAoZnVzZWQsIGdsb2JhbCwgbG9jYWwpIHNpZ21vaWQgWzEyXSBlYWNoOyB0YWJsZSAoTVMxLXN0eWxlIHNsaWNlcywgcm93cywgY29scywgd2VpZ2h0cyBbMTAsIFRPUF0pIG92ZXJyaWRlcwogICAgc2VsZi1sb2NhdGlvbiAodGVzdCBob29rKS4iIiIKICAgIGNzID0gdW5pZm9ybV9jZW50cmVzKG1hc2spCiAgICBsbyA9IChJTUcgLSBFVkFMX1JFUykgLy8gMgogICAgeGcgPSB3aW5kb3dzKHZvbCwgY3MsIGRldmljZSlbLi4uLCBsbzpsbyArIEVWQUxfUkVTLCBsbzpsbyArIEVWQUxfUkVTXS5mbG9hdCgpLmRpdl8oMjU1LjApCiAgICB3aXRoIHRvcmNoLmF1dG9jYXN0KGRldmljZS50eXBlLCBkdHlwZT10b3JjaC5mbG9hdDE2LCBlbmFibGVkPWFtcCBhbmQgZGV2aWNlLnR5cGUgPT0gImN1ZGEiKToKICAgICAgICBmbSA9IG1vZGVsLmJhY2tib25lLmZvcndhcmRfZmVhdHVyZXMobm9ybWFsaXNlKHhnKSkKICAgIGZtID0gZm0uZmxvYXQoKQogICAgaCA9IG1vZGVsLm5vcm0uZmxvYXQoKShmbS5tZWFuKCgyLCAzKSkpCiAgICBhID0gdG9yY2guc29mdG1heChtb2RlbC5hdHQuZmxvYXQoKShoKSwgZGltPTApICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgW0ssIDEyXQogICAgZyA9ICgoYS5UIEAgaCkgKiBtb2RlbC5jbHNXLmZsb2F0KCkpLnN1bSgtMSkgKyBtb2RlbC5jbHNiLmZsb2F0KCkKICAgIFMsIFIsIFEsIEEgPSB0YWJsZSBpZiB0YWJsZSBpcyBub3QgTm9uZSBlbHNlIGxvY2F0ZShtb2RlbCwgZm0sIGEsIGNzKQogICAgZmlkID0gbnAucmVwZWF0KENGLCBUT1ApCiAgICBTLCBSLCBRID0gUy5yYXZlbCgpLCBSLnJhdmVsKCksIFEucmF2ZWwoKQogICAgaGIgPSBCT1hOW2ZpZF0gLy8gMgogICAgUiwgUSA9IG5wLmNsaXAoUiwgaGIsIElNRyAtIGhiKSwgbnAuY2xpcChRLCBoYiwgSU1HIC0gaGIpCiAgICB3dCA9IChBIC8gQS5zdW0oMSwga2VlcGRpbXM9VHJ1ZSkpLnJhdmVsKCkKICAgIHhjID0gd2luZG93cyh2b2wsIFMsIGRldmljZSkuZmxvYXQoKS5kaXZfKDI1NS4wKQogICAgcyA9IHRvcmNoLmFzX3RlbnNvcihCT1hOW2ZpZF0gLyBJTUcsIGR0eXBlPXRvcmNoLmZsb2F0MzIsIGRldmljZT1kZXZpY2UpLnZpZXcoLTEsIDEsIDEpCiAgICBweHkgPSB0b3JjaC5zdGFjayhbdG9yY2guYXNfdGVuc29yKChRICsgMC41KSAvIElNRyAqIDIgLSAxKSwgdG9yY2guYXNfdGVuc29yKChSICsgMC41KSAvIElNRyAqIDIgLSAxKV0sIDEpCiAgICB0aCA9IHRvcmNoLmNhdChbcyAqIHRvcmNoLmV5ZSgyLCBkZXZpY2U9ZGV2aWNlKS5leHBhbmQobGVuKGZpZCksIDIsIDIpLCBweHkuZmxvYXQoKS50byhkZXZpY2UpWy4uLiwgTm9uZV1dLCAyKQogICAgeGMgPSBGLmdyaWRfc2FtcGxlKHhjLCBGLmFmZmluZV9ncmlkKHRoLCAobGVuKGZpZCksIDMsIENST1BfUkVTLCBDUk9QX1JFUyksIGFsaWduX2Nvcm5lcnM9RmFsc2UpLCBtb2RlPSJiaWxpbmVhciIsCiAgICAgICAgICAgICAgICAgICAgICAgcGFkZGluZ19tb2RlPSJ6ZXJvcyIsIGFsaWduX2Nvcm5lcnM9RmFsc2UpCiAgICB3aXRoIHRvcmNoLmF1dG9jYXN0KGRldmljZS50eXBlLCBkdHlwZT10b3JjaC5mbG9hdDE2LCBlbmFibGVkPWFtcCBhbmQgZGV2aWNlLnR5cGUgPT0gImN1ZGEiKToKICAgICAgICBoYyA9IG1vZGVsLmNyb3BfZmVhdHVyZXMobm9ybWFsaXNlKHhjKSkKICAgIGhjID0gbW9kZWwubm9ybV9sLmZsb2F0KCkoaGMuZmxvYXQoKSkKICAgIGZpZHQgPSB0b3JjaC5hc190ZW5zb3IoZmlkLCBkZXZpY2U9ZGV2aWNlKQogICAgbGMgPSAoaGMgKiBtb2RlbC5jbHNWLmZsb2F0KClbZmlkdF0pLnN1bSgtMSkgKyBtb2RlbC5jbHN2Yi5mbG9hdCgpW2ZpZHRdCiAgICBsID0gdG9yY2guemVyb3MoMTIsIGRldmljZT1kZXZpY2UpLnNjYXR0ZXJfYWRkKDAsIGZpZHQsIGxjICogdG9yY2guYXNfdGVuc29yKHd0LCBkZXZpY2U9ZGV2aWNlKSkKICAgIGNtID0gdG9yY2guYXNfdGVuc29yKENNQVNLLCBkZXZpY2U9ZGV2aWNlKQogICAgZnVzZWQgPSB0b3JjaC53aGVyZShjbSwgMC41ICogKGcgKyBsKSwgZykKICAgIHJldHVybiB0b3JjaC5zdGFjayhbZnVzZWQsIGcsIHRvcmNoLndoZXJlKGNtLCBsLCBnKV0pLnNpZ21vaWQoKS5jcHUoKS5udW1weSgpICAgIyBsb2NhbDogZyB3aGVyZSBubyBjcm9wcwoKCmRlZiBlbnN1cmVfdGltbSgpOgogICAgdHJ5OgogICAgICAgIGltcG9ydCB0aW1tCiAgICAgICAgaWYgdGltbS5fX3ZlcnNpb25fXyA9PSAiMS4wLjI5IjoKICAgICAgICAgICAgcmV0dXJuCiAgICBleGNlcHQgSW1wb3J0RXJyb3I6CiAgICAgICAgcGFzcwogICAgYXJ0ID0gZGF0YXNldF9kaXIoInR1bWlzMDAxL3JzbmEtY29hdG5ldC1kOTYtZXhwMWItZm9sZDAiKSAgICAgICAgICAgICAgICAgIyBob2xkcyB0aGUgdGltbSAxLjAuMjkgd2hlZWwKICAgIHN1YnByb2Nlc3MucnVuKFtzeXMuZXhlY3V0YWJsZSwgIi1tIiwgInBpcCIsICJpbnN0YWxsIiwgIi0tbm8tZGVwcyIsICItLW5vLWluZGV4IiwgIi1xIiwKICAgICAgICAgICAgICAgICAgICBzdHIoYXJ0IC8gInRpbW0tMS4wLjI5LXB5My1ub25lLWFueS53aGwiKV0sIGNoZWNrPVRydWUpCgoKZGVmIG1haW4oKToKICAgIHN0YXJ0ZWQgPSB0aW1lLm1vbm90b25pYygpCiAgICBkYXRhLCB3b3JrID0gY29tcGV0aXRpb25fZGlyKCksIFBhdGgoIi9rYWdnbGUvd29ya2luZyIpCiAgICBzYW1wbGUgPSByZWFkX2NzdihkYXRhIC8gInNhbXBsZV9zdWJtaXNzaW9uLmNzdiIpCiAgICBpZiBub3Qgc2FtcGxlIG9yIGxpc3Qoc2FtcGxlWzBdKSAhPSBbIlN0dWR5SW5zdGFuY2VVSUQiLCAqVEFSR0VUU106CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic2FtcGxlX3N1Ym1pc3Npb24uY3N2IGNvbHVtbnMgZGlmZmVyIGZyb20gdGhlIG1vZGVsIHRhcmdldCBvcmRlciIpCiAgICBpZHMgPSBbclsiU3R1ZHlJbnN0YW5jZVVJRCJdIGZvciByIGluIHNhbXBsZV0KICAgIGZhbGxiYWNrID0ge3JbIlN0dWR5SW5zdGFuY2VVSUQiXTogW2Zsb2F0KHJbdF0pIGZvciB0IGluIFRBUkdFVFNdIGZvciByIGluIHNhbXBsZX0KCiAgICBkZWYgd3JpdGUocHJlZHMpOgogICAgICAgIHdpdGggKHdvcmsgLyAic3VibWlzc2lvbi5jc3YiKS5vcGVuKCJ3IiwgbmV3bGluZT0iIiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgdyA9IGNzdi53cml0ZXIoZikKICAgICAgICAgICAgdy53cml0ZXJvdyhbIlN0dWR5SW5zdGFuY2VVSUQiLCAqVEFSR0VUU10pCiAgICAgICAgICAgIGZvciBzIGluIGlkczoKICAgICAgICAgICAgICAgIHcud3JpdGVyb3coW3MsICpbZiJ7djouOGZ9IiBmb3IgdiBpbiBwcmVkc1tzXV1dKQogICAgd3JpdGUoZmFsbGJhY2spICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIyB2YWxpZCBmaWxlIGV2ZW4gaWYgaW5mZXJlbmNlIGlzIGludGVycnVwdGVkCgogICAgZW5zdXJlX3RpbW0oKQogICAgZGV2aWNlID0gdG9yY2guZGV2aWNlKCJjdWRhIiBpZiB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpIGVsc2UgImNwdSIpCiAgICBtb2RlbCA9IGxvYWRfbW9kZWwoZGF0YXNldF9kaXIoV0VJR0hUU1siZGF0YXNldCJdKSAvIFdFSUdIVFNbImZpbGUiXSwgZGV2aWNlLCBXRUlHSFRTWyJzaGEyNTYiXSkKICAgIHByaW50KGYiU3R1ZGllcyB7bGVuKGlkcyl9IHwgZGV2aWNlIHtkZXZpY2V9IHwgdjQgbDRjb3B5LCBzZWxmLWxvY2F0ZWQgY3JvcHMiLCBmbHVzaD1UcnVlKQogICAgYnlfc3R1ZHkgPSB7fQogICAgZm9yIHJvdyBpbiByZWFkX2NzdihkYXRhIC8gInRlc3Rfc2VyaWVzLmNzdiIpOgogICAgICAgIGJ5X3N0dWR5LnNldGRlZmF1bHQocm93WyJTdHVkeUluc3RhbmNlVUlEIl0sIFtdKS5hcHBlbmQocm93KQogICAgcHJlZHMsIGhlYWRzID0ge30sIHt9CiAgICBzdGF0cyA9IHsic3R1ZGllcyI6IGxlbihpZHMpLCAiZW1wdHlfc3R1ZGllcyI6IDAsICJkZWNvZGVfZXJyb3JzIjogMCwgImZwMzJfcmV0cmllcyI6IDB9CiAgICB3aXRoIFBvb2woNCkgYXMgcHJvY3MsIHRvcmNoLm5vX2dyYWQoKToKICAgICAgICBmb3IgbiwgKHNpZCwgdm9sLCBtYXNrLCBlcnJvcnMpIGluIGVudW1lcmF0ZSgKICAgICAgICAgICAgICAgIHByb2NzLmltYXAoYnVpbGRfc3R1ZHksIFsocywgYnlfc3R1ZHkuZ2V0KHMsIFtdKSwgZGF0YSAvICJ0ZXN0X3NlcmllcyIpIGZvciBzIGluIGlkc10pLCAxKToKICAgICAgICAgICAgc3RhdHNbImRlY29kZV9lcnJvcnMiXSArPSBlcnJvcnMKICAgICAgICAgICAgaWYgbm90IG1hc2suYW55KCk6CiAgICAgICAgICAgICAgICBzdGF0c1siZW1wdHlfc3R1ZGllcyJdICs9IDEKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHAgPSBwcmVkaWN0X3N0dWR5KG1vZGVsLCB2b2wsIG1hc2ssIGRldmljZSkKICAgICAgICAgICAgaWYgbm90IG5wLmlzZmluaXRlKHApLmFsbCgpOgogICAgICAgICAgICAgICAgc3RhdHNbImZwMzJfcmV0cmllcyJdICs9IDEKICAgICAgICAgICAgICAgIHAgPSBwcmVkaWN0X3N0dWR5KG1vZGVsLCB2b2wsIG1hc2ssIGRldmljZSwgYW1wPUZhbHNlKQogICAgICAgICAgICBpZiBwLnNoYXBlICE9ICgzLCAxMikgb3Igbm90IG5wLmlzZmluaXRlKHApLmFsbCgpOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIkludmFsaWQgcHJlZGljdGlvbiB7c2lkfSIpCiAgICAgICAgICAgIHByZWRzW3NpZF0gPSBwWzBdLnRvbGlzdCgpOyBoZWFkc1tzaWRdID0geyJnbG9iYWwiOiBwWzFdLnRvbGlzdCgpLCAibG9jYWwiOiBwWzJdLnRvbGlzdCgpfQogICAgICAgICAgICBpZiBuICUgNTAgPT0gMCBvciBuID09IGxlbihpZHMpOgogICAgICAgICAgICAgICAgcHJpbnQoZiJ2NCB7bn0ve2xlbihpZHMpfSBzdHVkaWVzOyB7dGltZS5tb25vdG9uaWMoKSAtIHN0YXJ0ZWQ6LjBmfXMiLCBmbHVzaD1UcnVlKQogICAgZmluYWwgPSB7czogcHJlZHMuZ2V0KHMsIGZhbGxiYWNrW3NdKSBmb3IgcyBpbiBpZHN9CiAgICB3cml0ZShmaW5hbCkKICAgIHdyaXR0ZW4gPSByZWFkX2Nzdih3b3JrIC8gInN1Ym1pc3Npb24uY3N2IikKICAgIHZhbHMgPSBucC5hcnJheShbW2Zsb2F0KHJbdF0pIGZvciB0IGluIFRBUkdFVFNdIGZvciByIGluIHdyaXR0ZW5dKQogICAgaWYgW3JbIlN0dWR5SW5zdGFuY2VVSUQiXSBmb3IgciBpbiB3cml0dGVuXSAhPSBpZHMgb3Igbm90IG5wLmlzZmluaXRlKHZhbHMpLmFsbCgpIG9yIHZhbHMubWluKCkgPCAwIG9yIHZhbHMubWF4KCkgPiAxOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlNhdmVkIHN1Ym1pc3Npb24gaXMgaW52YWxpZCIpCiAgICAod29yayAvICJoZWFkX3ByZWRpY3Rpb25zLmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoaGVhZHMpICsgIlxuIikKICAgIHN1bW1hcnkgPSB7Im1vZGVsIjogIlBVQi1LRC1FIEU0OiByNTAtR0wgdjQgbDRjb3B5IHMyMDI2IFNXQSAyMS0yNSBvbiBmdWxsLWJsZW5kIG9yZGVyIHdpdGggUmFwdG9yIHRhcmdldCB2YWx1ZXMgKG1peDczOiAwLjcgdGVhY2hlciArIDAuMyBsYWJlbC12MiksIHNlbGYtbG9jYXRlZCBjcm9wcyIsICJ3ZWlnaHRzX3NoYTI1NiI6IFdFSUdIVFNbInNoYTI1NiJdLAogICAgICAgICAgICAgICAic3RhdHMiOiBzdGF0cywgInNlY29uZHMiOiByb3VuZCh0aW1lLm1vbm90b25pYygpIC0gc3RhcnRlZCwgMSl9CiAgICAod29yayAvICJpbmZlcmVuY2Vfc3VtbWFyeS5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKHN1bW1hcnksIGluZGVudD0yKSArICJcbiIpCiAgICBwcmludChqc29uLmR1bXBzKHN1bW1hcnksIGluZGVudD0yKSwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg=="}
SHA = {"v4": "2a570689d07a3a01caa96ca058076a7d695181848033a9e9dcd0f4b3a68d063b", "ra": "5107e5479e02c64e6bf239361b5fc577ea62fa6622e1e8c897c99fe39318bb3c", "B": "f04a1220efb7f8353c51389751c562bc0a6d3d5f5f1ef1691513556ea7cf4f2f", "E4": "080cf63f8997a768d0b6290e71a597c53d7be02c30564df47eee951c20398a36", "raptor384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"}
DS = {"v4": "rsna-v4-l4copy-fold0", "ra": "rsna-pubkd-r-ra-fold0", "B": "rsna-pubkd-kd-b-fold0", "E4": "rsna-pubkd-kd-e4-fold0"}
CASES = {"ACL": ["1.2.826.0.1.3680043.8.498.11719502398697390809777045891822044637", "1.2.826.0.1.3680043.8.498.11804799183537166337413388415687675758"], "PF OA": ["1.2.826.0.1.3680043.8.498.13210084125872700841685814144002495145", "1.2.826.0.1.3680043.8.498.33045687849853291066101186962073873042"], "Fracture": ["1.2.826.0.1.3680043.8.498.88711356400820562340756013729440417015", "1.2.826.0.1.3680043.8.498.80931292574057303692838531323879513417"], "Baker's": ["1.2.826.0.1.3680043.8.498.18114567737638963627123562895002602285", "1.2.826.0.1.3680043.8.498.11268635423021790546063903457498005204"], "MCL": ["1.2.826.0.1.3680043.8.498.80878057166752747059190481131529937715", "1.2.826.0.1.3680043.8.498.42967970121147528831983176261453235744"], "Lateral OA": ["1.2.826.0.1.3680043.8.498.11958117172819754862618612170999909480", "1.2.826.0.1.3680043.8.498.12917504242141830416624247604384969664"], "Lateral Meniscus": ["1.2.826.0.1.3680043.8.498.96587029379314618196381307882235433083", "1.2.826.0.1.3680043.8.498.41619164749467663852546208143469380415"]}
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))
import kd_sub as sub  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

N1, N2, DRAWS = int(os.environ.get("KD_N1", 825)), int(os.environ.get("KD_N2", 200)), 4
RES, IMG, K = 320, sub.IMG, sub.EVAL_WINDOWS
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
AUG = {"mirror_p": 0.5, "rot_deg": 10.0, "scale": 0.10, "shift": 0.06, "gamma": 0.25, "gain": 0.10}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find(owner, slug):
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def checked(path, sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f"weights hash mismatch: {path}")
    return str(path)


class RaptorClassifier(nn.Module):
    """Their head (scripts/pubkd_raptor_teacher.py)."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0)
        return (torch.einsum("kn,kf->nf", a.to(h.dtype), h) * self.clsW).sum(-1) + self.clsb


def load_teacher(device):
    import timm
    p = sorted(find("nartaa", "rsna-knee-publication-swa-weights-20261007").rglob(
        "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"))
    assert len(p) == 1, p
    ck = torch.load(checked(p[0], SHA["raptor384"]), map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(sub.TARGETS), ck["lab"]
    bb = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False, num_classes=0, img_size=320)
    m = RaptorClassifier(bb, bb.num_features); m.load_state_dict(ck["model"], strict=True)
    for q in m.parameters():
        q.requires_grad_(False)
    return m.eval().to(device)


class Wrap:
    """Uniform access: feature maps -> pooled -> head, for the teacher (CoAtNet) and the r50 students."""

    def __init__(self, m, teacher):
        self.m, self.teacher = m, teacher

    def fmaps(self, x):                                  # x normalised [k, 3, R, R] -> fp32 [k, C, c, c]
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            fm = torch.cat([self.m.backbone.forward_features(c) for c in x.split(16)])
        return fm.float()

    def pool(self, fm):
        if self.teacher:
            return self.m.backbone.forward_head(fm, pre_logits=True)
        return fm.mean((2, 3))

    def head(self, fm):
        h = self.m.norm(self.pool(fm))
        al = self.m.att(h)
        a = torch.softmax(al, 0)
        z = ((a.T @ h) * self.m.clsW).sum(-1) + self.m.clsb
        return h, al, a, z


def norm_x(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def view(vol, cs, device, aug=False, whole=False, gen=None):
    """uint8 windows -> normalised [k, 3, 320, 320]. Clean: 320 centre crop (eval / target view). aug / whole: the
    runner's prepare() geometry on the whole 384 field (one draw per study), output 320 for the teacher's fixed size."""
    x = sub.windows(vol, cs, device)
    if aug and torch.rand(1, generator=gen).item() < AUG["mirror_p"]:
        sag = torch.as_tensor(slot_of(cs) <= 1, device=device)
        x = torch.where(sag.view(-1, 1, 1, 1), x.flip(1), x.flip(-1))
    x = x.float().div_(255.0)
    if not (aug or whole):
        lo = (IMG - RES) // 2
        return norm_x(x[..., lo:lo + RES, lo:lo + RES])
    u = lambda: (torch.rand(1, generator=gen).item() * 2 - 1)
    ang = u() * math.radians(AUG["rot_deg"]) if aug else 0.0
    sc = 1 + u() * AUG["scale"] if aug else 1.0
    tx, ty = (u() * AUG["shift"] * 2, u() * AUG["shift"] * 2) if aug else (0.0, 0.0)
    c, s = math.cos(ang) / sc, math.sin(ang) / sc
    theta = torch.tensor([[c, -s, tx], [s, c, ty]], device=device, dtype=torch.float32).expand(len(cs), 2, 3).contiguous()
    grid = F.affine_grid(theta, (len(cs), 3, RES, RES), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if aug:
        gamma, gain = math.exp(u() * AUG["gamma"]), 1 + u() * AUG["gain"]
        x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    return norm_x(x)


def train_centres(mask, k, rng):
    """coatnet_d96_runner.centres(mask, k, train=True): one random centre per 1/k of the valid range."""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, sub.MAXS - 2))])
    edges = np.linspace(0, len(cs), k + 1)
    idx = np.minimum((edges[:-1] + rng.random(k) * (edges[1:] - edges[:-1])).astype(int), len(cs) - 1)
    return np.clip(cs[idx], 1, sub.MAXS - 2)


def series_ablation(w, h, al, cs):
    sl = slot_of(cs); drop = np.full((5, 12), np.nan, np.float32); only = drop.copy()
    for j in range(5):
        for keep, out in ((sl != j, drop), (sl == j, only)):
            if keep.sum() == 0:
                continue
            kk = torch.as_tensor(keep, device=h.device)
            aa = torch.softmax(al[kk], 0)
            out[j] = (((aa.T @ h[kk]) * w.m.clsW).sum(-1) + w.m.clsb).detach().cpu().numpy()
    return drop, only


def cams(w, x, a, wins):
    """gradient x activation of finding f's contribution a[k, f] * (norm(pool(fm_k)) . clsW[f]) on window wins[f]."""
    out = np.zeros((12, 10, 10), np.float32)
    for k in sorted(set(int(v) for v in wins)):
        fs = [f for f in range(12) if int(wins[f]) == k]
        with torch.enable_grad():
            t = w.fmaps(x[k:k + 1]).detach().requires_grad_(True)
            hk = w.m.norm(w.pool(t))[0]
            for i, f in enumerate(fs):
                contrib = a[k, f].detach() * (hk @ w.m.clsW[f])
                g, = torch.autograd.grad(contrib, t, retain_graph=i < len(fs) - 1)
                cam = (g * t.detach()).sum(1)[0].clamp_min(0)
                if cam.shape[-1] != 10:
                    cam = F.adaptive_avg_pool2d(cam[None, None], 10)[0, 0]
                out[f] = cam.cpu().numpy()
    return out


def panel(vol, cs, f, k, cam_t, cam_s, pt, ps, path, title):
    from PIL import Image, ImageDraw
    lo = (IMG - RES) // 2
    img = vol[cs[k]][lo:lo + RES, lo:lo + RES].astype(np.float32)
    tiles = []
    for cam in (None, cam_t, cam_s):
        rgb = np.stack([img] * 3, -1)
        if cam is not None and cam.max() > 0:
            hm = np.kron(cam / cam.max(), np.ones((32, 32)))
            rgb[..., 0] = np.clip(rgb[..., 0] * 0.6 + 255 * hm * 0.8, 0, 255)
            rgb[..., 1] *= 1 - 0.5 * hm; rgb[..., 2] *= 1 - 0.5 * hm
        tiles.append(Image.fromarray(rgb.astype(np.uint8)))
    canvas = Image.new("RGB", (RES * 3, RES + 40), "black")
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * RES, 40))
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title, fill="white")
    d.text((4, 20), f"window {k} (slice {cs[k]}, slot {slot_of(cs[k])}) | image | teacher CAM p={pt:.2f} | E4 CAM p={ps:.2f}",
           fill="white")
    canvas.save(path)


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"teacher": Wrap(load_teacher(device), True)}
    for tag in ("v4", "ra", "B", "E4"):
        p = checked(find("tumis001", DS[tag]) / "weights_swa21_25.pt", SHA[tag])
        models[tag] = Wrap(sub.load_model(p, device).to(memory_format=torch.contiguous_format), False)
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    fold0 = json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())[:N1]
    rest = sorted(set(by_study) - set(json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())))
    train = [rest[i] for i in np.random.default_rng(2026).choice(len(rest), N2, replace=False)]
    case_of = {s: f for f, lst in CASES.items() for s in lst}
    log = {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "weights_sha256": SHA}
    tags = list(models)
    P1 = {"ids": [], "cs": []}
    for t in tags:
        for k in ("prob", "att", "drop", "only", "cam_t", "cam_own", "top"):
            P1[f"{t}_{k}"] = []
    (W / "panels").mkdir(exist_ok=True)
    errors = 0
    with Pool(4) as procs:
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in fold0])
        for n, (sid, vol, mask, err) in enumerate(it):
            errors += err
            cs = sub.uniform_centres(mask)
            x = view(vol, cs, device)
            res = {}
            with torch.no_grad():
                for t, w in models.items():
                    fm = w.fmaps(x); h, al, a, z = w.head(fm)
                    res[t] = (h, al, a, z)
                    if n == 0 and t == "teacher":
                        z_ref = w.m(x).float()
                        log["teacher_head_check_max_abs_logit_diff"] = float((z_ref - z).abs().max())
                    del fm
            kt = res["teacher"][2].argmax(0).cpu().numpy()                      # teacher's top window per finding
            P1["ids"].append(sid); P1["cs"].append(cs)
            for t, w in models.items():
                h, al, a, z = res[t]
                own = a.argmax(0).cpu().numpy()
                d_, o_ = series_ablation(w, h, al, cs)
                P1[f"{t}_prob"].append(torch.sigmoid(z).cpu().numpy()); P1[f"{t}_att"].append(a.cpu().numpy().astype(np.float16))
                P1[f"{t}_drop"].append(d_); P1[f"{t}_only"].append(o_); P1[f"{t}_top"].append(own)
                P1[f"{t}_cam_t"].append(cams(w, x, a, kt).astype(np.float16))
                P1[f"{t}_cam_own"].append(cams(w, x, a, own).astype(np.float16))
            if sid in case_of:
                f = list(sub.TARGETS).index(case_of[sid])
                panel(vol, cs, f, int(kt[f]), P1["teacher_cam_t"][-1][f].astype(np.float32),
                      P1["E4_cam_t"][-1][f].astype(np.float32), P1["teacher_prob"][-1][f], P1["E4_prob"][-1][f],
                      W / "panels" / f"{case_of[sid].replace(' ', '_').replace(chr(39), '')}_{sid[-8:]}.png",
                      f"{case_of[sid]} {sid[-12:]}: teacher-right / E4-wrong case")
            if n % 50 == 0:
                print(f"part 1: {n + 1}/{len(fold0)} {time.monotonic() - t0:.0f}s", flush=True)
    np.savez_compressed(W / "kd_gap_part1.npz", **{k: np.array(v) for k, v in P1.items()})
    log["part1_s"] = round(time.monotonic() - t0, 1); log["decode_errors_part1"] = errors

    rng = np.random.default_rng(7); gen = torch.Generator().manual_seed(7)
    tm = models["teacher"]
    P2 = {k: [] for k in ("ids", "clean", "sub16", "sub16aug", "all94aug", "whole", "cover16", "ra_clean", "E4_clean")}
    with Pool(4) as procs, torch.no_grad():
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in train])
        for n, (sid, vol, mask, err) in enumerate(it):
            cs = sub.uniform_centres(mask)
            x = view(vol, cs, device)
            _, _, a, z = tm.head(tm.fmaps(x))
            pr = lambda xx: torch.sigmoid(tm.head(tm.fmaps(xx))[3]).cpu().numpy()
            s16, s16a, cov = [], [], []
            for _ in range(DRAWS):
                c16 = train_centres(mask, 16, rng)
                s16.append(pr(view(vol, c16, device)))
                s16a.append(pr(view(vol, c16, device, aug=True, gen=gen)))
                near = (np.abs(cs[:, None] - c16[None, :]) <= 1).any(1)               # uniform windows within 1 slice
                cov.append(a.cpu().numpy()[near].sum(0))
            P2["ids"].append(sid); P2["clean"].append(torch.sigmoid(z).cpu().numpy())
            P2["sub16"].append(s16); P2["sub16aug"].append(s16a); P2["cover16"].append(cov)
            P2["all94aug"].append([pr(view(vol, cs, device, aug=True, gen=gen)) for _ in range(2)])
            P2["whole"].append(pr(view(vol, cs, device, whole=True)))
            for t in ("ra", "E4"):
                P2[f"{t}_clean"].append(torch.sigmoid(models[t].head(models[t].fmaps(x))[3]).cpu().numpy())
            if n % 25 == 0:
                print(f"part 2: {n + 1}/{len(train)} {time.monotonic() - t0:.0f}s", flush=True)
    np.savez_compressed(W / "kd_gap_part2.npz", **{k: np.array(v) for k, v in P2.items()})
    log["seconds"] = round(time.monotonic() - t0, 1)
    (W / "kd_gap_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()

"""KD-VIEW sweep (no training, no submission). Built by scripts/build_kaggle_kd_view_sweep.py; kd_sub = the E4
submission script: DICOM builder, windows, r50 l4copy model."""
import base64
import hashlib
import json
import math
import sys
import threading
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EMB = {"kd_sub.py": "IiIicjUwLUdMIHY0IGw0Y29weSBzMjAyNiBzaW5nbGUtbW9kZWwgc3VibWlzc2lvbiAobGVkZ2VyIHJvd3MgcjUwLUdMNCwgU0wtMHY0KS4gRm9sZCAwIChsYWJlbC12MiBjb25maWRlbnQgY2VsbHMpOgpmdXNlZCAwLjk0NzUgdnMgbGFiZWwtdjIgcjUwIDAuOTQyNSAoTEIgMC45MzcpLgoKT25lIFJlc05ldC01MCByZWFkcyBlYWNoIHN0dWR5IHR3aWNlOgogIGdsb2JhbCAgdGhlIDk0IGV2YWwgd2luZG93cyBvdmVyIHRoZSB2YWxpZCByYW5nZSwgMzIwIGNlbnRyZSBjcm9wIChsYWJlbC12MiAvIEUzIHBhdGgpIC0+IHdpbmRvdyBhdHRlbnRpb24gaGVhZCBnOwogIGxvY2FsICAgZm9yIGVhY2ggb2YgdGhlIDEwIGNyb3BwZWQgZmluZGluZ3MsIDQgem9vbWVkIGNyb3BzICgxOTIgcHggYm94LCAxMjggcHggZm9yIG1lZGlhbCBtZW5pc2N1cyAvIGNvbnR1c2lvbiwKICAgICAgICAgIHJlc2FtcGxlZCB0byAzODQpIHJ1biB0aHJvdWdoIHRoZSBzaGFyZWQgc3RlbS1sYXllcjMgKGNyb3AgQk4gc3RhdGlzdGljcykgYW5kIHRoZSBjcm9wcycgb3duIGxheWVyNCAtPgogICAgICAgICAgbG9jYWwgaGVhZCBsLCB3ZWlnaHRlZCBieSB0aGUgd2luZG93IGF0dGVudGlvbiB3aXRoaW4gdGhlIGZpbmRpbmcuCiAgZnVzZWQgICBzaWdtb2lkKChnICsgbCkgLyAyKSBvbiB0aGUgMTAgY3JvcHBlZCBmaW5kaW5ncywgc2lnbW9pZChnKSBvbiBlZmZ1c2lvbiAvIHN5bm92aXRpcy4KQ3JvcCBwbGFjZW1lbnQgaXMgc2VsZi1sb2NhdGVkIChTTC0wdjQpLCBubyBNUzE6IHBlciBmaW5kaW5nIHRoZSA0IHdpbmRvd3Mgd2l0aCB0aGUgbGFyZ2VzdCBnbG9iYWwgYXR0ZW50aW9uLCBvbiBlYWNoIHRoZQpncmFkaWVudCB4IGFjdGl2YXRpb24gY2VudHJvaWQgb2YgdGhhdCB3aW5kb3cncyBsb2dpdCBjb250cmlidXRpb24gb3ZlciB0aGUgMTAgeCAxMCBsYXllci00IGNlbGxzLgpWb2x1bWU6IEUzIGtlcm5lbCdzIGQ5NiBidWlsZGVyLCB1bmNoYW5nZWQuIGZwMTYgYXV0b2Nhc3Qgd2l0aCBhbiBmcDMyIHJldHJ5IGZvciBub24tZmluaXRlIGxvZ2l0czsgYSBzdHVkeSB0aGF0CmNhbm5vdCBiZSByZWFkIGtlZXBzIHRoZSBzYW1wbGVfc3VibWlzc2lvbiB2YWx1ZXMuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgY29weQppbXBvcnQgY3N2CmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCBzeXMKaW1wb3J0IHRpbWUKZnJvbSBjb25jdXJyZW50LmZ1dHVyZXMgaW1wb3J0IFRocmVhZFBvb2xFeGVjdXRvcgpmcm9tIG11bHRpcHJvY2Vzc2luZyBpbXBvcnQgUG9vbApmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgdG9yY2gKaW1wb3J0IHRvcmNoLm5uIGFzIG5uCmltcG9ydCB0b3JjaC5ubi5mdW5jdGlvbmFsIGFzIEYKClRBUkdFVFMgPSBbIkFDTCIsICJNQ0wiLCAiTWVkaWFsIE1lbmlzY3VzIiwgIkxhdGVyYWwgTWVuaXNjdXMiLCAiTWVkaWFsIE9BIiwgIkxhdGVyYWwgT0EiLAogICAgICAgICAgICJQRiBPQSIsICJFZmZ1c2lvbiIsICJTeW5vdml0aXMiLCAiQmFrZXIncyIsICJDb250dXNpb24iLCAiRnJhY3R1cmUiXQpJTUcsIENST1BfTU0sIEVWQUxfUkVTLCBFVkFMX1dJTkRPV1MgPSAzODQsIDE0MC4wLCAzMjAsIDk0ClNMT1RTID0gWygiU2FnaXR0YWwiLCAxLCAyNiksICgiU2FnaXR0YWwiLCAwLCAyMiksICgiQ29yb25hbCIsIDEsIDE4KSwgKCJDb3JvbmFsIiwgMCwgMTIpLCAoIkF4aWFsIiwgLTEsIDE4KV0KTUFYUyA9IHN1bShrIGZvciBfLCBfLCBrIGluIFNMT1RTKQpCT1ggPSB7IkFDTCI6IDE5MiwgIk1DTCI6IDE5MiwgIk1lZGlhbCBNZW5pc2N1cyI6IDEyOCwgIkxhdGVyYWwgTWVuaXNjdXMiOiAxOTIsICJNZWRpYWwgT0EiOiAxOTIsICJMYXRlcmFsIE9BIjogMTkyLAogICAgICAgIlBGIE9BIjogMTkyLCAiQmFrZXIncyI6IDE5MiwgIkNvbnR1c2lvbiI6IDEyOCwgIkZyYWN0dXJlIjogMTkyfQpDRiA9IG5wLmFycmF5KFtrIGZvciBrLCB0IGluIGVudW1lcmF0ZShUQVJHRVRTKSBpZiB0IGluIEJPWF0pCkJPWE4gPSBucC5hcnJheShbQk9YLmdldCh0LCAwKSBmb3IgdCBpbiBUQVJHRVRTXSkKQ01BU0sgPSBucC5pc2luKG5wLmFyYW5nZSgxMiksIENGKQpUT1AsIENST1BfUkVTID0gNCwgMzg0CldFSUdIVFMgPSB7ImRhdGFzZXQiOiAidHVtaXMwMDEvcnNuYS1wdWJrZC1rZC1lNC1mb2xkMCIsICJmaWxlIjogIndlaWdodHNfc3dhMjFfMjUucHQiLAogICAgICAgICAgICJzaGEyNTYiOiAiMDgwY2Y2M2Y4OTk3YTc2OGQwYjYyOTBlNzFhNTk3YzUzZDdiZTAyYzMwNTY0ZGY0N2VlZTk1MWMyMDM5OGEzNiJ9Ck1FQU4gPSB0b3JjaC50ZW5zb3IoWzAuNDg1LCAwLjQ1NiwgMC40MDZdKS52aWV3KDEsIDMsIDEsIDEpClNURCA9IHRvcmNoLnRlbnNvcihbMC4yMjksIDAuMjI0LCAwLjIyNV0pLnZpZXcoMSwgMywgMSwgMSkKCgpkZWYgY29tcGV0aXRpb25fZGlyKCk6CiAgICByZXR1cm4gbmV4dChwIGZvciBwIGluIChQYXRoKCIva2FnZ2xlL2lucHV0L2NvbXBldGl0aW9ucy9yc25hLWtuZWUtYWJub3JtYWxpdHktZGV0ZWN0aW9uIiksCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBQYXRoKCIva2FnZ2xlL2lucHV0L3JzbmEta25lZS1hYm5vcm1hbGl0eS1kZXRlY3Rpb24iKSkgaWYgcC5leGlzdHMoKSkKCgpkZWYgZGF0YXNldF9kaXIocmVmKToKICAgIG93bmVyLCBzbHVnID0gcmVmLnNwbGl0KCIvIikKICAgIHJldHVybiBuZXh0KHAgZm9yIHAgaW4gKFBhdGgoIi9rYWdnbGUvaW5wdXQvZGF0YXNldHMiKSAvIG93bmVyIC8gc2x1ZywgUGF0aCgiL2thZ2dsZS9pbnB1dCIpIC8gc2x1ZykgaWYgcC5leGlzdHMoKSkKCgpkZWYgcmVhZF9jc3YocGF0aCk6CiAgICB3aXRoIHBhdGgub3BlbihuZXdsaW5lPSIiLCBlbmNvZGluZz0idXRmLTgtc2lnIikgYXMgc3RyZWFtOgogICAgICAgIHJldHVybiBsaXN0KGNzdi5EaWN0UmVhZGVyKHN0cmVhbSkpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIGQ5NiBidWlsZGVyIChFMyBrZXJuZWwsIHVuY2hhbmdlZCkKZGVmIGhlYWRlcihwYXRoKToKICAgIGltcG9ydCBweWRpY29tCiAgICB0cnk6CiAgICAgICAgaCA9IHB5ZGljb20uZGNtcmVhZChwYXRoLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSkKICAgICAgICBpb3AgPSBnZXRhdHRyKGgsICJJbWFnZU9yaWVudGF0aW9uUGF0aWVudCIsIE5vbmUpCiAgICAgICAgaXBwID0gZ2V0YXR0cihoLCAiSW1hZ2VQb3NpdGlvblBhdGllbnQiLCBOb25lKQogICAgICAgIGlmIGlvcCBpcyBub3QgTm9uZSBhbmQgaXBwIGlzIG5vdCBOb25lIGFuZCBsZW4oaW9wKSA9PSA2OgogICAgICAgICAgICByLCBjID0gbnAuYXJyYXkoaW9wWzozXSwgZmxvYXQpLCBucC5hcnJheShpb3BbMzpdLCBmbG9hdCkKICAgICAgICAgICAgcG9zID0gZmxvYXQobnAuZG90KG5wLmFycmF5KGlwcCwgZmxvYXQpLCBucC5jcm9zcyhyLCBjKSkpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcG9zID0gZmxvYXQoZ2V0YXR0cihoLCAiSW5zdGFuY2VOdW1iZXIiLCAwKSBvciAwKQogICAgICAgIHBzID0gZ2V0YXR0cihoLCAiUGl4ZWxTcGFjaW5nIiwgTm9uZSkKICAgICAgICByZXR1cm4gcG9zLCBwYXRoLCAoZmxvYXQocHNbMF0pIGlmIHBzIGlzIG5vdCBOb25lIGVsc2UgMC41KSwgVHJ1ZQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gMC4wLCBwYXRoLCAwLjUsIEZhbHNlCgoKZGVmIHBpeGVscyhwYXRoKToKICAgIGltcG9ydCBweWRpY29tCiAgICBmcm9tIHB5ZGljb20ucGl4ZWxzIGltcG9ydCBhcHBseV9tb2RhbGl0eV9sdXQKICAgIGQgPSBweWRpY29tLmRjbXJlYWQocGF0aCkKICAgIGEgPSBhcHBseV9tb2RhbGl0eV9sdXQoZC5waXhlbF9hcnJheSwgZCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICBpZiBzdHIoZ2V0YXR0cihkLCAiUGhvdG9tZXRyaWNJbnRlcnByZXRhdGlvbiIsICIiKSkgPT0gIk1PTk9DSFJPTUUxIjoKICAgICAgICBhID0gYS5tYXgoKSAtIGEKICAgIHJldHVybiBhCgoKZGVmIGNyb3BfcmVzaXplKGEsIHBzKToKICAgIGltcG9ydCBjdjIKICAgIGgsIHcgPSBhLnNoYXBlCiAgICBjcHggPSBtaW4oaW50KHJvdW5kKENST1BfTU0gLyBtYXgocHMsIDFlLTMpKSksIGgsIHcpCiAgICB5MCwgeDAgPSAoaCAtIGNweCkgLy8gMiwgKHcgLSBjcHgpIC8vIDIKICAgIHJldHVybiBjdjIucmVzaXplKGFbeTA6eTAgKyBjcHgsIHgwOngwICsgY3B4XSwgKElNRywgSU1HKSwgaW50ZXJwb2xhdGlvbj1jdjIuSU5URVJfQVJFQSkKCgpkZWYgYnVpbGRfc3R1ZHkoam9iKToKICAgIHN0dWR5X2lkLCByb3dzLCBzZXJpZXNfZGlyID0gam9iICAgICAgICAgICAgICAgICAgICAgIyBzZXJpZXNfZGlyOiA8Y29tcGV0aXRpb24+L3Rlc3Rfc2VyaWVzICh0cmFpbiBmb3IgY2hlY2tzKQogICAgdm9sID0gbnAuemVyb3MoKE1BWFMsIElNRywgSU1HKSwgbnAudWludDgpCiAgICBpZHgsIHVzZWQsIGVycm9ycyA9IDAsIHNldCgpLCAwCiAgICB3aXRoIFRocmVhZFBvb2xFeGVjdXRvcihtYXhfd29ya2Vycz04KSBhcyBwb29sOgogICAgICAgIGZvciBwbGFuZSwgZmx1aWQsIGsgaW4gU0xPVFM6CiAgICAgICAgICAgIGNhbmRzID0gW3IgZm9yIHIgaW4gcm93cyBpZiByWyJBbmF0b21pY2FsX1BsYW5lIl0gPT0gcGxhbmUgYW5kIHJbIlNlcmllc0luc3RhbmNlVUlEIl0gbm90IGluIHVzZWRdCiAgICAgICAgICAgIHByZWYgPSBbciBmb3IgciBpbiBjYW5kcyBpZiBmbHVpZCBpbiAoMCwgMSkgYW5kIGludChyLmdldCgiRmx1aWRfU2Vuc2l0aXZlIiwgMCkgb3IgMCkgPT0gZmx1aWRdCiAgICAgICAgICAgIHJvdyA9IChwcmVmIG9yIGNhbmRzIG9yIFtOb25lXSlbMF0KICAgICAgICAgICAgaWYgcm93IGlzIE5vbmU6CiAgICAgICAgICAgICAgICBpZHggKz0gawogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgdXNlZC5hZGQocm93WyJTZXJpZXNJbnN0YW5jZVVJRCJdKQogICAgICAgICAgICBmaWxlcyA9IHNvcnRlZCgoc2VyaWVzX2RpciAvIHN0dWR5X2lkIC8gcm93WyJTZXJpZXNJbnN0YW5jZVVJRCJdKS5nbG9iKCIqLmRjbSIpKQogICAgICAgICAgICBoZWFkcyA9IGxpc3QocG9vbC5tYXAoaGVhZGVyLCBmaWxlcykpCiAgICAgICAgICAgIG9rX3BzID0gW2hbMl0gZm9yIGggaW4gaGVhZHMgaWYgaFszXV0KICAgICAgICAgICAgbWVkX3BzID0gZmxvYXQobnAubWVkaWFuKG9rX3BzKSkgaWYgb2tfcHMgZWxzZSAwLjUKICAgICAgICAgICAgaGVhZHMuc29ydChrZXk9bGFtYmRhIGg6IGhbMF0pCiAgICAgICAgICAgIG4gPSBsZW4oaGVhZHMpCiAgICAgICAgICAgIGlmIG4gPT0gMDoKICAgICAgICAgICAgICAgIGlkeCArPSBrCiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBsbywgaGkgPSBpbnQobiAqIDAuMDIpLCBpbnQobiAqIDAuOTgpIC0gMQogICAgICAgICAgICBoaSA9IG1heChoaSwgbG8pCiAgICAgICAgICAgIHBpY2tzID0gW2ludChtaW4ocCwgbiAtIDEpKSBmb3IgcCBpbiAobnAubGluc3BhY2UobG8sIGhpLCBrKS5yb3VuZCgpLmFzdHlwZShpbnQpIGlmIG4gPiAxIGVsc2UgWzBdICogayldCgogICAgICAgICAgICBkZWYgbG9hZChwKToKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICByZXR1cm4gcGl4ZWxzKGhlYWRzW3BdWzFdKSwgaGVhZHNbcF1bMl0KICAgICAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIE5vbmUsIG1lZF9wcwogICAgICAgICAgICBnb3QgPSBsaXN0KHBvb2wubWFwKGxvYWQsIHBpY2tzKSkKICAgICAgICAgICAgdmFsaWQgPSBbYSBmb3IgYSwgXyBpbiBnb3QgaWYgYSBpcyBub3QgTm9uZV0KICAgICAgICAgICAgZXJyb3JzICs9IGxlbihnb3QpIC0gbGVuKHZhbGlkKQogICAgICAgICAgICBsb3EsIGhpcSA9IChucC5wZXJjZW50aWxlKG5wLmNvbmNhdGVuYXRlKFthLnJhdmVsKCkgZm9yIGEgaW4gdmFsaWRdKSwgWzIuMCwgOTguMF0pIGlmIHZhbGlkIGVsc2UgKDAuMCwgMS4wKSkKICAgICAgICAgICAgZm9yIGEsIHBzIGluIGdvdDoKICAgICAgICAgICAgICAgIGlmIGEgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICAgICAgYXcgPSBucC5jbGlwKChhIC0gbG9xKSAvIChoaXEgLSBsb3EgKyAxZS02KSwgMCwgMSkKICAgICAgICAgICAgICAgICAgICB2b2xbaWR4XSA9IChjcm9wX3Jlc2l6ZShhdywgcHMgaWYgcHMgPiAwIGVsc2UgbWVkX3BzKSAqIDI1NSkuYXN0eXBlKG5wLnVpbnQ4KQogICAgICAgICAgICAgICAgaWR4ICs9IDEKICAgIG1hc2sgPSB2b2wucmVzaGFwZShNQVhTLCAtMSkubWF4KDEpID4gMAogICAgcmV0dXJuIHN0dWR5X2lkLCB2b2wsIG1hc2ssIGVycm9ycwoKCmRlZiB1bmlmb3JtX2NlbnRyZXMobWFzayk6CiAgICAiIiI5NCBjZW50cmVzIG92ZXIgdGhlIHdob2xlIHZhbGlkIHJhbmdlIChjb2F0bmV0X2Q5Nl9ydW5uZXIuY2VudHJlcywgZXZhbCksIGNsaXBwZWQgdG8gWzEsIE1BWFMgLSAyXS4iIiIKICAgIHZhbGlkID0gbnAud2hlcmUobWFzaylbMF0KICAgIGlmIGxlbih2YWxpZCkgPCAzOgogICAgICAgIHZhbGlkID0gbnAuYXJhbmdlKDMpCiAgICBsbywgaGkgPSBpbnQodmFsaWQubWluKCkpLCBpbnQodmFsaWQubWF4KCkpCiAgICBjcyA9IG5wLmFyYW5nZShsbyArIDEsIGhpKSBpZiBoaSAtIGxvID49IDIgZWxzZSBucC5hcnJheShbbWF4KDEsIG1pbigobG8gKyBoaSkgLy8gMiwgTUFYUyAtIDIpKV0pCiAgICByZXR1cm4gbnAuY2xpcChjc1tucC5saW5zcGFjZSgwLCBsZW4oY3MpIC0gMSwgRVZBTF9XSU5ET1dTKS5yb3VuZCgpLmFzdHlwZShpbnQpXSwgMSwgTUFYUyAtIDIpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIG1vZGVsIChyNTBfZ2w0X3J1bm5lci5SYXB0b3JHTDQsIGFybSBsNGNvcHkpCmNsYXNzIER1YWxCTihubi5CYXRjaE5vcm0yZCk6CiAgICAiIiJCYXRjaE5vcm0yZCB3aXRoIGEgc2Vjb25kIHNldCBvZiBydW5uaW5nIHN0YXRpc3RpY3MgZm9yIGNyb3AgaW1hZ2VzIChjcm9wID0gVHJ1ZSk7IGFmZmluZSBzaGFyZWQuIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGJuKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKGJuLm51bV9mZWF0dXJlcywgYm4uZXBzLCBibi5tb21lbnR1bSwgYm4uYWZmaW5lLCBibi50cmFja19ydW5uaW5nX3N0YXRzKQogICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCJjcm9wX21lYW4iLCB0b3JjaC56ZXJvcyhibi5udW1fZmVhdHVyZXMpKQogICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCJjcm9wX3ZhciIsIHRvcmNoLm9uZXMoYm4ubnVtX2ZlYXR1cmVzKSkKICAgICAgICBzZWxmLnJlZ2lzdGVyX2J1ZmZlcigiY3JvcF9uYnQiLCB0b3JjaC56ZXJvcygoKSwgZHR5cGU9dG9yY2gubG9uZykpCiAgICAgICAgc2VsZi5jcm9wID0gRmFsc2UKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB4KToKICAgICAgICBpZiBub3Qgc2VsZi5jcm9wOgogICAgICAgICAgICByZXR1cm4gc3VwZXIoKS5mb3J3YXJkKHgpCiAgICAgICAgcmV0dXJuIEYuYmF0Y2hfbm9ybSh4LCBzZWxmLmNyb3BfbWVhbiwgc2VsZi5jcm9wX3Zhciwgc2VsZi53ZWlnaHQsIHNlbGYuYmlhcywgRmFsc2UsIDAuMCwgc2VsZi5lcHMpCgoKZGVmIHdyYXBfYm5zKG1vZHVsZSk6CiAgICBvdXQgPSBbXQogICAgZm9yIG5hbWUsIGNoIGluIG1vZHVsZS5uYW1lZF9jaGlsZHJlbigpOgogICAgICAgIGlmIGlzaW5zdGFuY2UoY2gsIG5uLkJhdGNoTm9ybTJkKToKICAgICAgICAgICAgZCA9IER1YWxCTihjaCk7IHNldGF0dHIobW9kdWxlLCBuYW1lLCBkKTsgb3V0LmFwcGVuZChkKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIG91dCArPSB3cmFwX2JucyhjaCkKICAgIHJldHVybiBvdXQKCgpjbGFzcyBWNEw0Q29weShubi5Nb2R1bGUpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIGltcG9ydCB0aW1tCiAgICAgICAgc2VsZi5iYWNrYm9uZSA9IHRpbW0uY3JlYXRlX21vZGVsKCJyZXNuZXQ1MC50dl9pbjFrIiwgcHJldHJhaW5lZD1GYWxzZSwgbnVtX2NsYXNzZXM9MCkKICAgICAgICBkID0gc2VsZi5iYWNrYm9uZS5udW1fZmVhdHVyZXMKICAgICAgICBzZWxmLm5vcm0gPSBubi5MYXllck5vcm0oZCkKICAgICAgICBzZWxmLmF0dCA9IG5uLlNlcXVlbnRpYWwobm4uTGluZWFyKGQsIDI1NiksIG5uLlRhbmgoKSwgbm4uRHJvcG91dCgwLjIpLCBubi5MaW5lYXIoMjU2LCAxMikpCiAgICAgICAgc2VsZi5jbHNXID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKDEyLCBkKSk7IHNlbGYuY2xzYiA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcygxMikpCiAgICAgICAgc2VsZi5ibnMgPSB3cmFwX2JucyhzZWxmLmJhY2tib25lKQogICAgICAgIGFzc2VydCBsZW4oc2VsZi5ibnMpID09IDUzCiAgICAgICAgc2VsZi5sYXllcjRfYyA9IGNvcHkuZGVlcGNvcHkoc2VsZi5iYWNrYm9uZS5sYXllcjQpCiAgICAgICAgZm9yIG0gaW4gc2VsZi5sYXllcjRfYy5tb2R1bGVzKCk6CiAgICAgICAgICAgIGlmIGlzaW5zdGFuY2UobSwgRHVhbEJOKToKICAgICAgICAgICAgICAgIG0uY3JvcCA9IFRydWUKICAgICAgICBzZWxmLm5vcm1fbCA9IG5uLkxheWVyTm9ybShkKQogICAgICAgIHNlbGYuY2xzViA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcygxMiwgZCkpOyBzZWxmLmNsc3ZiID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKDEyKSkKCiAgICBkZWYgY3JvcF9mZWF0dXJlcyhzZWxmLCB4KToKICAgICAgICBiYiA9IHNlbGYuYmFja2JvbmUKICAgICAgICBmb3IgbSBpbiBzZWxmLmJuczoKICAgICAgICAgICAgbS5jcm9wID0gVHJ1ZQogICAgICAgIHRyeToKICAgICAgICAgICAgaCA9IGJiLm1heHBvb2woYmIuYWN0MShiYi5ibjEoYmIuY29udjEoeCkpKSkKICAgICAgICAgICAgaCA9IHNlbGYubGF5ZXI0X2MoYmIubGF5ZXIzKGJiLmxheWVyMihiYi5sYXllcjEoaCkpKSkKICAgICAgICAgICAgcmV0dXJuIGJiLmZvcndhcmRfaGVhZChoKQogICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgIGZvciBtIGluIHNlbGYuYm5zOgogICAgICAgICAgICAgICAgbS5jcm9wID0gRmFsc2UKCgpkZWYgbG9hZF9tb2RlbChwYXRoLCBkZXZpY2UsIHNoYTI1Nj1Ob25lKToKICAgIGlmIHNoYTI1NiBpcyBub3QgTm9uZSBhbmQgaGFzaGxpYi5zaGEyNTYoUGF0aChwYXRoKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpICE9IHNoYTI1NjoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJXZWlnaHRzIGhhc2ggbWlzbWF0Y2g6IHY0IGw0Y29weSIpCiAgICBtb2RlbCA9IFY0TDRDb3B5KCkKICAgIG1vZGVsLmxvYWRfc3RhdGVfZGljdCh0b3JjaC5sb2FkKHBhdGgsIG1hcF9sb2NhdGlvbj0iY3B1Iiwgd2VpZ2h0c19vbmx5PVRydWUpLCBzdHJpY3Q9VHJ1ZSkKICAgIG1vZGVsID0gbW9kZWwudG8oZGV2aWNlKS5ldmFsKCkudG8obWVtb3J5X2Zvcm1hdD10b3JjaC5jaGFubmVsc19sYXN0KQogICAgZm9yIHAgaW4gbW9kZWwucGFyYW1ldGVycygpOgogICAgICAgIHAucmVxdWlyZXNfZ3JhZF8oRmFsc2UpCiAgICByZXR1cm4gbW9kZWwKCgpkZWYgd2luZG93cyh2b2wsIGNzLCBkZXZpY2UpOgogICAgIiIiW2tdIGNlbnRyZSBzbGljZXMgLT4gdWludDggW2ssIDMsIElNRywgSU1HXSBvbiBkZXZpY2UgKHNsaWNlcyBjIC0gMSwgYywgYyArIDEpLiIiIgogICAgcmV0dXJuIHRvcmNoLmZyb21fbnVtcHkobnAuYXNjb250aWd1b3VzYXJyYXkodm9sW25wLnN0YWNrKFtjcyAtIDEsIGNzLCBjcyArIDFdLCAxKV0pKS50byhkZXZpY2UpCgoKZGVmIG5vcm1hbGlzZSh4KToKICAgIHJldHVybiAoKHggLSBNRUFOLnRvKHguZGV2aWNlKSkgLyBTVEQudG8oeC5kZXZpY2UpKS5jb250aWd1b3VzKG1lbW9yeV9mb3JtYXQ9dG9yY2guY2hhbm5lbHNfbGFzdCkKCgpkZWYgbG9jYXRlKG1vZGVsLCBmbSwgYSwgY3MpOgogICAgIiIic2VsZmxvY192NF9ncHUubG9jYXRlIG9uIHRoZSBnbG9iYWwgcGFzczogZm0gW0ssIDIwNDgsIGMsIGNdIGZwMzIsIGEgW0ssIDEyXSAtPiAoc2xpY2VzLCByb3dzLCBjb2xzLCB3ZWlnaHRzKQogICAgWzEwLCBUT1BdIGZvciB0aGUgY3JvcHBlZCBmaW5kaW5ncy4iIiIKICAgIGRldiA9IGZtLmRldmljZQogICAgVyA9IG1vZGVsLmNsc1cuZmxvYXQoKVtDRl0KICAgIGFjID0gYVs6LCBDRl0KICAgIHRvcCA9IGFjLnRvcGsoVE9QLCBkaW09MCkuaW5kaWNlcyAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIyBbVE9QLCAxMF0KICAgIFUgPSB0b3JjaC51bmlxdWUodG9wKTsgcm93ID0ge2ludChrKTogaiBmb3IgaiwgayBpbiBlbnVtZXJhdGUoVS50b2xpc3QoKSl9CiAgICB3aXRoIHRvcmNoLmVuYWJsZV9ncmFkKCk6CiAgICAgICAgdCA9IGZtW1VdLmRldGFjaCgpLmNsb25lKCkucmVxdWlyZXNfZ3JhZF8oVHJ1ZSkKICAgICAgICBjb250cmliID0gYWNbVV0gKiAobW9kZWwubm9ybS5mbG9hdCgpKHQubWVhbigoMiwgMykpKSBAIFcuVCkgICAgICAgICAgIyBbdSwgMTBdLCBhIGZpeGVkCiAgICAgICAgYyA9IHQuc2hhcGVbLTFdOyBweCA9IEVWQUxfUkVTIC8gYzsgZ2kgPSB0b3JjaC5hcmFuZ2UoYywgZGV2aWNlPWRldiwgZHR5cGU9dG9yY2guZmxvYXQzMikKICAgICAgICBsbyA9IChJTUcgLSBFVkFMX1JFUykgLy8gMgogICAgICAgIFMgPSBucC56ZXJvcygobGVuKENGKSwgVE9QKSwgbnAuaW50NjQpOyBSID0gUy5jb3B5KCk7IFEgPSBTLmNvcHkoKTsgQSA9IG5wLnplcm9zKChsZW4oQ0YpLCBUT1ApLCBucC5mbG9hdDMyKQogICAgICAgIGZvciBqIGluIHJhbmdlKGxlbihDRikpOgogICAgICAgICAgICBnciwgPSB0b3JjaC5hdXRvZ3JhZC5ncmFkKGNvbnRyaWJbOiwgal0uc3VtKCksIHQsIHJldGFpbl9ncmFwaD1qIDwgbGVuKENGKSAtIDEpCiAgICAgICAgICAgIGtzID0gdG9wWzosIGpdLnRvbGlzdCgpOyByb3dzID0gW3Jvd1trXSBmb3IgayBpbiBrc10KICAgICAgICAgICAgbXAgPSAoZ3Jbcm93c10gKiB0LmRldGFjaCgpW3Jvd3NdKS5zdW0oMSkuY2xhbXBfbWluKDApCiAgICAgICAgICAgIHRvdCA9IG1wLnN1bSgoMSwgMikpOyBvayA9IHRvdCA+IDA7IHRvdCA9IHRvdC5jbGFtcF9taW4oMWUtMTIpCiAgICAgICAgICAgIGN5ID0gdG9yY2gud2hlcmUob2ssIChtcC5zdW0oMikgKiBnaSkuc3VtKDEpIC8gdG90LCB0b3JjaC5mdWxsX2xpa2UodG90LCAoYyAtIDEpIC8gMikpCiAgICAgICAgICAgIGN4ID0gdG9yY2gud2hlcmUob2ssIChtcC5zdW0oMSkgKiBnaSkuc3VtKDEpIC8gdG90LCB0b3JjaC5mdWxsX2xpa2UodG90LCAoYyAtIDEpIC8gMikpCiAgICAgICAgICAgIFNbal0gPSBjc1trc10KICAgICAgICAgICAgUltqXSA9IG5wLnJvdW5kKChsbyArIChjeSArIDAuNSkgKiBweCkuY3B1KCkubnVtcHkoKSkuYXN0eXBlKG5wLmludDY0KQogICAgICAgICAgICBRW2pdID0gbnAucm91bmQoKGxvICsgKGN4ICsgMC41KSAqIHB4KS5jcHUoKS5udW1weSgpKS5hc3R5cGUobnAuaW50NjQpCiAgICAgICAgICAgIEFbal0gPSBhY1t0b3BbOiwgal0sIGpdLmRldGFjaCgpLmNwdSgpLm51bXB5KCkKICAgIHJldHVybiBTLCBSLCBRLCBBCgoKZGVmIHByZWRpY3Rfc3R1ZHkobW9kZWwsIHZvbCwgbWFzaywgZGV2aWNlLCBhbXA9VHJ1ZSwgdGFibGU9Tm9uZSk6CiAgICAiIiItPiAoZnVzZWQsIGdsb2JhbCwgbG9jYWwpIHNpZ21vaWQgWzEyXSBlYWNoOyB0YWJsZSAoTVMxLXN0eWxlIHNsaWNlcywgcm93cywgY29scywgd2VpZ2h0cyBbMTAsIFRPUF0pIG92ZXJyaWRlcwogICAgc2VsZi1sb2NhdGlvbiAodGVzdCBob29rKS4iIiIKICAgIGNzID0gdW5pZm9ybV9jZW50cmVzKG1hc2spCiAgICBsbyA9IChJTUcgLSBFVkFMX1JFUykgLy8gMgogICAgeGcgPSB3aW5kb3dzKHZvbCwgY3MsIGRldmljZSlbLi4uLCBsbzpsbyArIEVWQUxfUkVTLCBsbzpsbyArIEVWQUxfUkVTXS5mbG9hdCgpLmRpdl8oMjU1LjApCiAgICB3aXRoIHRvcmNoLmF1dG9jYXN0KGRldmljZS50eXBlLCBkdHlwZT10b3JjaC5mbG9hdDE2LCBlbmFibGVkPWFtcCBhbmQgZGV2aWNlLnR5cGUgPT0gImN1ZGEiKToKICAgICAgICBmbSA9IG1vZGVsLmJhY2tib25lLmZvcndhcmRfZmVhdHVyZXMobm9ybWFsaXNlKHhnKSkKICAgIGZtID0gZm0uZmxvYXQoKQogICAgaCA9IG1vZGVsLm5vcm0uZmxvYXQoKShmbS5tZWFuKCgyLCAzKSkpCiAgICBhID0gdG9yY2guc29mdG1heChtb2RlbC5hdHQuZmxvYXQoKShoKSwgZGltPTApICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgW0ssIDEyXQogICAgZyA9ICgoYS5UIEAgaCkgKiBtb2RlbC5jbHNXLmZsb2F0KCkpLnN1bSgtMSkgKyBtb2RlbC5jbHNiLmZsb2F0KCkKICAgIFMsIFIsIFEsIEEgPSB0YWJsZSBpZiB0YWJsZSBpcyBub3QgTm9uZSBlbHNlIGxvY2F0ZShtb2RlbCwgZm0sIGEsIGNzKQogICAgZmlkID0gbnAucmVwZWF0KENGLCBUT1ApCiAgICBTLCBSLCBRID0gUy5yYXZlbCgpLCBSLnJhdmVsKCksIFEucmF2ZWwoKQogICAgaGIgPSBCT1hOW2ZpZF0gLy8gMgogICAgUiwgUSA9IG5wLmNsaXAoUiwgaGIsIElNRyAtIGhiKSwgbnAuY2xpcChRLCBoYiwgSU1HIC0gaGIpCiAgICB3dCA9IChBIC8gQS5zdW0oMSwga2VlcGRpbXM9VHJ1ZSkpLnJhdmVsKCkKICAgIHhjID0gd2luZG93cyh2b2wsIFMsIGRldmljZSkuZmxvYXQoKS5kaXZfKDI1NS4wKQogICAgcyA9IHRvcmNoLmFzX3RlbnNvcihCT1hOW2ZpZF0gLyBJTUcsIGR0eXBlPXRvcmNoLmZsb2F0MzIsIGRldmljZT1kZXZpY2UpLnZpZXcoLTEsIDEsIDEpCiAgICBweHkgPSB0b3JjaC5zdGFjayhbdG9yY2guYXNfdGVuc29yKChRICsgMC41KSAvIElNRyAqIDIgLSAxKSwgdG9yY2guYXNfdGVuc29yKChSICsgMC41KSAvIElNRyAqIDIgLSAxKV0sIDEpCiAgICB0aCA9IHRvcmNoLmNhdChbcyAqIHRvcmNoLmV5ZSgyLCBkZXZpY2U9ZGV2aWNlKS5leHBhbmQobGVuKGZpZCksIDIsIDIpLCBweHkuZmxvYXQoKS50byhkZXZpY2UpWy4uLiwgTm9uZV1dLCAyKQogICAgeGMgPSBGLmdyaWRfc2FtcGxlKHhjLCBGLmFmZmluZV9ncmlkKHRoLCAobGVuKGZpZCksIDMsIENST1BfUkVTLCBDUk9QX1JFUyksIGFsaWduX2Nvcm5lcnM9RmFsc2UpLCBtb2RlPSJiaWxpbmVhciIsCiAgICAgICAgICAgICAgICAgICAgICAgcGFkZGluZ19tb2RlPSJ6ZXJvcyIsIGFsaWduX2Nvcm5lcnM9RmFsc2UpCiAgICB3aXRoIHRvcmNoLmF1dG9jYXN0KGRldmljZS50eXBlLCBkdHlwZT10b3JjaC5mbG9hdDE2LCBlbmFibGVkPWFtcCBhbmQgZGV2aWNlLnR5cGUgPT0gImN1ZGEiKToKICAgICAgICBoYyA9IG1vZGVsLmNyb3BfZmVhdHVyZXMobm9ybWFsaXNlKHhjKSkKICAgIGhjID0gbW9kZWwubm9ybV9sLmZsb2F0KCkoaGMuZmxvYXQoKSkKICAgIGZpZHQgPSB0b3JjaC5hc190ZW5zb3IoZmlkLCBkZXZpY2U9ZGV2aWNlKQogICAgbGMgPSAoaGMgKiBtb2RlbC5jbHNWLmZsb2F0KClbZmlkdF0pLnN1bSgtMSkgKyBtb2RlbC5jbHN2Yi5mbG9hdCgpW2ZpZHRdCiAgICBsID0gdG9yY2guemVyb3MoMTIsIGRldmljZT1kZXZpY2UpLnNjYXR0ZXJfYWRkKDAsIGZpZHQsIGxjICogdG9yY2guYXNfdGVuc29yKHd0LCBkZXZpY2U9ZGV2aWNlKSkKICAgIGNtID0gdG9yY2guYXNfdGVuc29yKENNQVNLLCBkZXZpY2U9ZGV2aWNlKQogICAgZnVzZWQgPSB0b3JjaC53aGVyZShjbSwgMC41ICogKGcgKyBsKSwgZykKICAgIHJldHVybiB0b3JjaC5zdGFjayhbZnVzZWQsIGcsIHRvcmNoLndoZXJlKGNtLCBsLCBnKV0pLnNpZ21vaWQoKS5jcHUoKS5udW1weSgpICAgIyBsb2NhbDogZyB3aGVyZSBubyBjcm9wcwoKCmRlZiBlbnN1cmVfdGltbSgpOgogICAgdHJ5OgogICAgICAgIGltcG9ydCB0aW1tCiAgICAgICAgaWYgdGltbS5fX3ZlcnNpb25fXyA9PSAiMS4wLjI5IjoKICAgICAgICAgICAgcmV0dXJuCiAgICBleGNlcHQgSW1wb3J0RXJyb3I6CiAgICAgICAgcGFzcwogICAgYXJ0ID0gZGF0YXNldF9kaXIoInR1bWlzMDAxL3JzbmEtY29hdG5ldC1kOTYtZXhwMWItZm9sZDAiKSAgICAgICAgICAgICAgICAgIyBob2xkcyB0aGUgdGltbSAxLjAuMjkgd2hlZWwKICAgIHN1YnByb2Nlc3MucnVuKFtzeXMuZXhlY3V0YWJsZSwgIi1tIiwgInBpcCIsICJpbnN0YWxsIiwgIi0tbm8tZGVwcyIsICItLW5vLWluZGV4IiwgIi1xIiwKICAgICAgICAgICAgICAgICAgICBzdHIoYXJ0IC8gInRpbW0tMS4wLjI5LXB5My1ub25lLWFueS53aGwiKV0sIGNoZWNrPVRydWUpCgoKZGVmIG1haW4oKToKICAgIHN0YXJ0ZWQgPSB0aW1lLm1vbm90b25pYygpCiAgICBkYXRhLCB3b3JrID0gY29tcGV0aXRpb25fZGlyKCksIFBhdGgoIi9rYWdnbGUvd29ya2luZyIpCiAgICBzYW1wbGUgPSByZWFkX2NzdihkYXRhIC8gInNhbXBsZV9zdWJtaXNzaW9uLmNzdiIpCiAgICBpZiBub3Qgc2FtcGxlIG9yIGxpc3Qoc2FtcGxlWzBdKSAhPSBbIlN0dWR5SW5zdGFuY2VVSUQiLCAqVEFSR0VUU106CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic2FtcGxlX3N1Ym1pc3Npb24uY3N2IGNvbHVtbnMgZGlmZmVyIGZyb20gdGhlIG1vZGVsIHRhcmdldCBvcmRlciIpCiAgICBpZHMgPSBbclsiU3R1ZHlJbnN0YW5jZVVJRCJdIGZvciByIGluIHNhbXBsZV0KICAgIGZhbGxiYWNrID0ge3JbIlN0dWR5SW5zdGFuY2VVSUQiXTogW2Zsb2F0KHJbdF0pIGZvciB0IGluIFRBUkdFVFNdIGZvciByIGluIHNhbXBsZX0KCiAgICBkZWYgd3JpdGUocHJlZHMpOgogICAgICAgIHdpdGggKHdvcmsgLyAic3VibWlzc2lvbi5jc3YiKS5vcGVuKCJ3IiwgbmV3bGluZT0iIiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgdyA9IGNzdi53cml0ZXIoZikKICAgICAgICAgICAgdy53cml0ZXJvdyhbIlN0dWR5SW5zdGFuY2VVSUQiLCAqVEFSR0VUU10pCiAgICAgICAgICAgIGZvciBzIGluIGlkczoKICAgICAgICAgICAgICAgIHcud3JpdGVyb3coW3MsICpbZiJ7djouOGZ9IiBmb3IgdiBpbiBwcmVkc1tzXV1dKQogICAgd3JpdGUoZmFsbGJhY2spICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIyB2YWxpZCBmaWxlIGV2ZW4gaWYgaW5mZXJlbmNlIGlzIGludGVycnVwdGVkCgogICAgZW5zdXJlX3RpbW0oKQogICAgZGV2aWNlID0gdG9yY2guZGV2aWNlKCJjdWRhIiBpZiB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpIGVsc2UgImNwdSIpCiAgICBtb2RlbCA9IGxvYWRfbW9kZWwoZGF0YXNldF9kaXIoV0VJR0hUU1siZGF0YXNldCJdKSAvIFdFSUdIVFNbImZpbGUiXSwgZGV2aWNlLCBXRUlHSFRTWyJzaGEyNTYiXSkKICAgIHByaW50KGYiU3R1ZGllcyB7bGVuKGlkcyl9IHwgZGV2aWNlIHtkZXZpY2V9IHwgdjQgbDRjb3B5LCBzZWxmLWxvY2F0ZWQgY3JvcHMiLCBmbHVzaD1UcnVlKQogICAgYnlfc3R1ZHkgPSB7fQogICAgZm9yIHJvdyBpbiByZWFkX2NzdihkYXRhIC8gInRlc3Rfc2VyaWVzLmNzdiIpOgogICAgICAgIGJ5X3N0dWR5LnNldGRlZmF1bHQocm93WyJTdHVkeUluc3RhbmNlVUlEIl0sIFtdKS5hcHBlbmQocm93KQogICAgcHJlZHMsIGhlYWRzID0ge30sIHt9CiAgICBzdGF0cyA9IHsic3R1ZGllcyI6IGxlbihpZHMpLCAiZW1wdHlfc3R1ZGllcyI6IDAsICJkZWNvZGVfZXJyb3JzIjogMCwgImZwMzJfcmV0cmllcyI6IDB9CiAgICB3aXRoIFBvb2woNCkgYXMgcHJvY3MsIHRvcmNoLm5vX2dyYWQoKToKICAgICAgICBmb3IgbiwgKHNpZCwgdm9sLCBtYXNrLCBlcnJvcnMpIGluIGVudW1lcmF0ZSgKICAgICAgICAgICAgICAgIHByb2NzLmltYXAoYnVpbGRfc3R1ZHksIFsocywgYnlfc3R1ZHkuZ2V0KHMsIFtdKSwgZGF0YSAvICJ0ZXN0X3NlcmllcyIpIGZvciBzIGluIGlkc10pLCAxKToKICAgICAgICAgICAgc3RhdHNbImRlY29kZV9lcnJvcnMiXSArPSBlcnJvcnMKICAgICAgICAgICAgaWYgbm90IG1hc2suYW55KCk6CiAgICAgICAgICAgICAgICBzdGF0c1siZW1wdHlfc3R1ZGllcyJdICs9IDEKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHAgPSBwcmVkaWN0X3N0dWR5KG1vZGVsLCB2b2wsIG1hc2ssIGRldmljZSkKICAgICAgICAgICAgaWYgbm90IG5wLmlzZmluaXRlKHApLmFsbCgpOgogICAgICAgICAgICAgICAgc3RhdHNbImZwMzJfcmV0cmllcyJdICs9IDEKICAgICAgICAgICAgICAgIHAgPSBwcmVkaWN0X3N0dWR5KG1vZGVsLCB2b2wsIG1hc2ssIGRldmljZSwgYW1wPUZhbHNlKQogICAgICAgICAgICBpZiBwLnNoYXBlICE9ICgzLCAxMikgb3Igbm90IG5wLmlzZmluaXRlKHApLmFsbCgpOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIkludmFsaWQgcHJlZGljdGlvbiB7c2lkfSIpCiAgICAgICAgICAgIHByZWRzW3NpZF0gPSBwWzBdLnRvbGlzdCgpOyBoZWFkc1tzaWRdID0geyJnbG9iYWwiOiBwWzFdLnRvbGlzdCgpLCAibG9jYWwiOiBwWzJdLnRvbGlzdCgpfQogICAgICAgICAgICBpZiBuICUgNTAgPT0gMCBvciBuID09IGxlbihpZHMpOgogICAgICAgICAgICAgICAgcHJpbnQoZiJ2NCB7bn0ve2xlbihpZHMpfSBzdHVkaWVzOyB7dGltZS5tb25vdG9uaWMoKSAtIHN0YXJ0ZWQ6LjBmfXMiLCBmbHVzaD1UcnVlKQogICAgZmluYWwgPSB7czogcHJlZHMuZ2V0KHMsIGZhbGxiYWNrW3NdKSBmb3IgcyBpbiBpZHN9CiAgICB3cml0ZShmaW5hbCkKICAgIHdyaXR0ZW4gPSByZWFkX2Nzdih3b3JrIC8gInN1Ym1pc3Npb24uY3N2IikKICAgIHZhbHMgPSBucC5hcnJheShbW2Zsb2F0KHJbdF0pIGZvciB0IGluIFRBUkdFVFNdIGZvciByIGluIHdyaXR0ZW5dKQogICAgaWYgW3JbIlN0dWR5SW5zdGFuY2VVSUQiXSBmb3IgciBpbiB3cml0dGVuXSAhPSBpZHMgb3Igbm90IG5wLmlzZmluaXRlKHZhbHMpLmFsbCgpIG9yIHZhbHMubWluKCkgPCAwIG9yIHZhbHMubWF4KCkgPiAxOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlNhdmVkIHN1Ym1pc3Npb24gaXMgaW52YWxpZCIpCiAgICAod29yayAvICJoZWFkX3ByZWRpY3Rpb25zLmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoaGVhZHMpICsgIlxuIikKICAgIHN1bW1hcnkgPSB7Im1vZGVsIjogIlBVQi1LRC1FIEU0OiByNTAtR0wgdjQgbDRjb3B5IHMyMDI2IFNXQSAyMS0yNSBvbiBmdWxsLWJsZW5kIG9yZGVyIHdpdGggUmFwdG9yIHRhcmdldCB2YWx1ZXMgKG1peDczOiAwLjcgdGVhY2hlciArIDAuMyBsYWJlbC12MiksIHNlbGYtbG9jYXRlZCBjcm9wcyIsICJ3ZWlnaHRzX3NoYTI1NiI6IFdFSUdIVFNbInNoYTI1NiJdLAogICAgICAgICAgICAgICAic3RhdHMiOiBzdGF0cywgInNlY29uZHMiOiByb3VuZCh0aW1lLm1vbm90b25pYygpIC0gc3RhcnRlZCwgMSl9CiAgICAod29yayAvICJpbmZlcmVuY2Vfc3VtbWFyeS5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKHN1bW1hcnksIGluZGVudD0yKSArICJcbiIpCiAgICBwcmludChqc29uLmR1bXBzKHN1bW1hcnksIGluZGVudD0yKSwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg=="}
SHA = {"E4": "080cf63f8997a768d0b6290e71a597c53d7be02c30564df47eee951c20398a36", "384": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef", "224": "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"}
N, BUDGET_S = 825, 10800
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))
import kd_sub as sub  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RES, IMG = 320, sub.IMG
BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18])
slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
Z = dict(mirror_p=0.0, crop=True, rot=0.0, scale=0.0, shift=0.0, gamma=0.0, gain=0.0)
AUGS = {"none": Z, "mirror": {**Z, "mirror_p": 1.0}, "fov": {**Z, "crop": False},
        "geom": {**Z, "crop": False, "rot": 10.0, "scale": 0.10, "shift": 0.06},
        "int": {**Z, "gamma": 0.25, "gain": 0.10},
        "full": dict(mirror_p=0.5, crop=False, rot=10.0, scale=0.10, shift=0.06, gamma=0.25, gain=0.10),
        "light": dict(mirror_p=0.0, crop=True, rot=5.0, scale=0.05, shift=0.03, gamma=0.10, gain=0.05)}
VIEWS = [("clean", "u", 94, "none"), ("r16", "r", 16, "none"), ("r32", "r", 32, "none"), ("r48", "r", 48, "none"),
         ("g16", "g", 16, "none"), ("g32", "g", 32, "none"),
         ("u_mirror", "u", 94, "mirror"), ("u_fov", "u", 94, "fov"), ("u_geom", "u", 94, "geom"),
         ("u_int", "u", 94, "int"), ("u_full", "u", 94, "full"), ("u_light", "u", 94, "light"),
         ("r16_full", "r", 16, "full"), ("r32_full", "r", 32, "full"), ("r48_full", "r", 48, "full"),
         ("r16_light", "r", 16, "light"), ("r32_light", "r", 32, "light"), ("r48_light", "r", 48, "light"),
         ("g16_light", "g", 16, "light"), ("g32_light", "g", 32, "light")]
COVER = ["r16", "r32", "r48", "g16", "g32"]


def find(owner, slug):
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


def checked(path, sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f"weights hash mismatch: {path}")
    return str(path)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))


def load_teacher(device, which):
    import timm
    name = {"384": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
            "224": "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"}[which]
    p = sorted(find("nartaa", "rsna-knee-publication-swa-weights-20261007").rglob(name))
    assert len(p) == 1, p
    ck = torch.load(checked(p[0], SHA[which]), map_location="cpu", weights_only=True)
    assert list(ck["lab"]) == list(sub.TARGETS), ck["lab"]
    bb = timm.create_model(f"coatnet_rmlp_2_rw_{which}", pretrained=False, num_classes=0,
                           **({"img_size": 320} if which == "384" else {}))
    m = RaptorClassifier(bb, bb.num_features); m.load_state_dict(ck["model"], strict=True)
    for q in m.parameters():
        q.requires_grad_(False)
    return m.eval().to(device)


class Wrap:
    def __init__(self, m, teacher):
        self.m, self.teacher = m, teacher

    def fmaps(self, x):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            fm = torch.cat([self.m.backbone.forward_features(c) for c in x.split(16)])
        return fm.float()

    def pool(self, fm):
        return self.m.backbone.forward_head(fm, pre_logits=True) if self.teacher else fm.mean((2, 3))

    def head(self, fm):
        h = self.m.norm(self.pool(fm))
        a = torch.softmax(self.m.att(h), 0)
        return a, ((a.T @ h) * self.m.clsW).sum(-1) + self.m.clsb

    def prob(self, x):
        return torch.sigmoid(self.head(self.fmaps(x))[1]).cpu().numpy()


def norm_x(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def view(vol, cs, device, mode, gen):
    """uint8 windows -> normalised [k, 3, 320, 320]; the clean / mirror / int crops are exact pixel slices."""
    A = AUGS[mode]; u = lambda: torch.rand(1, generator=gen).item() * 2 - 1
    x = sub.windows(vol, cs, device)
    if A["mirror_p"] > 0 and torch.rand(1, generator=gen).item() < A["mirror_p"]:
        sag = torch.as_tensor(slot_of(cs) <= 1, device=device)
        x = torch.where(sag.view(-1, 1, 1, 1), x.flip(1), x.flip(-1))
    x = x.float().div_(255.0)
    if A["crop"] and A["rot"] == 0 and A["scale"] == 0 and A["shift"] == 0:
        lo = (IMG - RES) // 2
        x = x[..., lo:lo + RES, lo:lo + RES]
    else:
        f = RES / IMG if A["crop"] else 1.0
        ang, sc = u() * math.radians(A["rot"]), 1 + u() * A["scale"]
        tx, ty = u() * A["shift"] * 2, u() * A["shift"] * 2
        c, s = math.cos(ang) / sc * f, math.sin(ang) / sc * f
        theta = torch.tensor([[c, -s, tx], [s, c, ty]], device=device, dtype=torch.float32).expand(len(cs), 2, 3).contiguous()
        x = F.grid_sample(x, F.affine_grid(theta, (len(cs), 3, RES, RES), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    if A["gamma"] > 0:
        gamma, gain = math.exp(u() * A["gamma"]), 1 + u() * A["gain"]
        x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    return norm_x(x)


def view224(vol, cs, device):
    """The 224 sibling's own eval view (scripts/pubkd_raptor_teacher.py): 0.8 centre crop (307 px) -> 224."""
    x = sub.windows(vol, cs, device).float().div_(255.0)
    c = int(round(IMG * 0.8)); o = (IMG - c) // 2
    return norm_x(F.interpolate(x[..., o:o + c, o:o + c], size=(224, 224), mode="bilinear", align_corners=False))


def train_centres(mask, k, rng):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, sub.MAXS - 2))])
    edges = np.linspace(0, len(cs), k + 1)
    idx = np.minimum((edges[:-1] + rng.random(k) * (edges[1:] - edges[:-1])).astype(int), len(cs) - 1)
    return np.clip(cs[idx], 1, sub.MAXS - 2)


def guided_centres(cs, att, k, rng):
    p = 0.5 * att.mean(1) / att.mean(1).sum() + 0.5 / len(cs)
    return cs[np.sort(rng.choice(len(cs), size=min(k, len(cs)), replace=False, p=p / p.sum()))]


def cam224_on(w, x, kt):
    """224 sibling's gradient x activation map for finding f on window kt[f], mapped to the 384's 10 x 10 grid."""
    out = np.zeros((12, 10, 10), np.float32)
    a, _ = w.head(w.fmaps(x))
    c = int(round(IMG * 0.8)); o = (IMG - c) // 2; lo = (IMG - RES) // 2; pl = o - lo; pr_ = RES - c - pl
    for k in sorted(set(int(v) for v in kt)):
        fs = [f for f in range(12) if int(kt[f]) == k]
        with torch.enable_grad():
            t = w.fmaps(x[k:k + 1]).detach().requires_grad_(True)
            hk = w.m.norm(w.pool(t))[0]
            for i, f in enumerate(fs):
                g, = torch.autograd.grad(a[k, f] * (hk @ w.m.clsW[f]), t, retain_graph=i < len(fs) - 1)
                cam = (g * t.detach()).sum(1).clamp_min(0)[None]
                cam = F.pad(F.interpolate(cam, size=(c, c), mode="bilinear", align_corners=False), (pl, pr_, pl, pr_))
                out[f] = F.avg_pool2d(cam, 32)[0, 0].cpu().numpy()
    return out


def worker(rank, device, ids, it, res, t0, log):
    torch.set_grad_enabled(False)
    try:
        T = Wrap(load_teacher(device, "384"), True); T2 = Wrap(load_teacher(device, "224"), True)
        p = checked(find("tumis001", "rsna-pubkd-kd-e4-fold0") / "weights_swa21_25.pt", SHA["E4"])
        S = Wrap(sub.load_model(p, device).to(memory_format=torch.contiguous_format), False)
        rng = np.random.default_rng(100 + rank); gen = torch.Generator().manual_seed(100 + rank)
        if True:
            for n, (sid, vol, mask, err) in enumerate(it):
                if time.monotonic() - t0 > BUDGET_S:
                    log[f"stopped_{rank}"] = n; break
                cs = sub.uniform_centres(mask)
                x = view(vol, cs, device, "none", gen)
                a, z = T.head(T.fmaps(x)); att = a.cpu().numpy(); kt = att.argmax(0)
                r = {"att": att.astype(np.float16), "top": kt, "err": err}
                for name, mode, k, aug in VIEWS:
                    if name == "clean":
                        c_ = cs; xv = x
                    else:
                        c_ = cs if mode == "u" else train_centres(mask, k, rng) if mode == "r" else guided_centres(cs, att, k, rng)
                        xv = view(vol, c_, device, aug, gen)
                    r[f"T_{name}"] = T.prob(xv) if name != "clean" else torch.sigmoid(z).cpu().numpy()
                    r[f"S_{name}"] = S.prob(xv)
                    if name in COVER:
                        r[f"cov_{name}"] = att[(np.abs(cs[:, None] - c_[None, :]) <= 1).any(1)].sum(0)
                x2 = view224(vol, cs, device)
                r["T224_clean"] = T2.prob(x2)
                r["cam224"] = cam224_on(T2, x2, kt).astype(np.float16)
                res[sid] = r
                if n % 25 == 0:
                    print(f"gpu {rank}: {n + 1}/{len(ids)} {time.monotonic() - t0:.0f}s", flush=True)
    except Exception:
        log[f"error_{rank}"] = traceback.format_exc(); print(log[f"error_{rank}"], flush=True)


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    fold0 = json.loads((find("tumis001", "rsna-d96-fold0-ids") / "fold0_ids.json").read_text())[:N]
    ng = max(1, torch.cuda.device_count())
    procs = Pool(4)                                   # forked before any CUDA context and before the worker threads
    devs = [torch.device(f"cuda:{i}") for i in range(ng)] if torch.cuda.is_available() else [torch.device("cpu")]
    log = {"devices": [torch.cuda.get_device_name(i) for i in range(ng)] if torch.cuda.is_available() else ["cpu"],
           "weights_sha256": SHA, "views": VIEWS, "augs": AUGS}
    res = {}
    shards = [fold0[i::len(devs)] for i in range(len(devs))]
    its = [procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in sh]) for sh in shards]
    th = [threading.Thread(target=worker, args=(i, d, shards[i], its[i], res, t0, log)) for i, d in enumerate(devs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    procs.terminate()
    ids = [s for s in fold0 if s in res]
    keys = list(res[ids[0]])
    np.savez_compressed(W / "kd_view_sweep.npz", ids=np.array(ids), **{k: np.array([res[s][k] for s in ids]) for k in keys})
    log["n_done"] = len(ids); log["seconds"] = round(time.monotonic() - t0, 1)
    log["s_per_study_per_gpu"] = round(log["seconds"] * len(devs) / max(1, len(ids)), 2)
    (W / "kd_view_sweep_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: v for k, v in log.items() if k not in ("views", "augs")}, default=str), flush=True)


if __name__ == "__main__":
    main()

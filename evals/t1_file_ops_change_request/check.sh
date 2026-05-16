python verify.py > /tmp/verify.out && \
grep -qx 'PASS' /tmp/verify.out && \
grep -qx 'PASS' verification.txt

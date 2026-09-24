echo "=== search fairface chunks ==="
find /home/research /mnt/d /mnt/c/Users/research -iname '*fairface*' 2>/dev/null | head -40
echo "=== C: free ==="
df -h /mnt/c 2>/dev/null | tail -1
echo "=== D: ==="
ls /mnt/d 2>/dev/null | head
echo "=== old e85 ==="
ls /mnt/c/Users/research/e85 2>/dev/null | head
ls /mnt/d/from-C/e85 2>/dev/null | head

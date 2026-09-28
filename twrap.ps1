Set-Location 'E:\Disaster prediction project'; python tapi.py *>&1 | Select-String -NotMatch 'INFO|WARNING' | Out-File -FilePath 'tapi.log' -Encoding utf8

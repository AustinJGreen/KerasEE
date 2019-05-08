powershell.exe -nologo -noprofile -command "& { Compress-Archive -Path .\eedream\*.* -CompressionLevel Optimal -DestinationPath .\eedream.zip }"
rd /s /q .\eedream\
if not exist .\eedream\ rd /s /q .\eedream\
exit
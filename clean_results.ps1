# remove-item except .gitignore
remove-item -path ".\result\test\*" -Recurse -exclude .gitignore
Write-Output "Cleaned results folder"

# remove-item except .gitignore
remove-item -path ".\result\latest\*" -Recurse -exclude .gitignore
Write-Output "Cleaned results folder"

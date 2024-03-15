chcp 65001 > NUL
@echo off
setlocal
echo 不要になった以下のフォルダ・ファイルを削除します:
echo 注: 学習やマージ等はApp.batへ統合されました。
echo Style-Bert-VITS2\common\
echo Style-Bert-VITS2\monotonic_align\
echo Style-Bert-VITS2\text\
echo Style-Bert-VITS2\tools\
echo Style-Bert-VITS2\attentions.py
echo Style-Bert-VITS2\commons.py
echo Style-Bert-VITS2\Dataset.bat
echo Style-Bert-VITS2\infer.py
echo Style-Bert-VITS2\Merge.bat
echo Style-Bert-VITS2\models_jp_extra.py
echo Style-Bert-VITS2\models.py
echo Style-Bert-VITS2\modules.py
echo Style-Bert-VITS2\re_matching.py
echo Style-Bert-VITS2\spec_gen.py
echo Style-Bert-VITS2\Style.bat
echo Style-Bert-VITS2\Train.bat
echo Style-Bert-VITS2\transforms.py
echo Style-Bert-VITS2\update_status.py
echo Style-Bert-VITS2\utils.py
echo Style-Bert-VITS2\webui_dataset.py
echo Style-Bert-VITS2\webui_merge.py
echo Style-Bert-VITS2\webui_style_vectors.py
echo Style-Bert-VITS2\webui_train.py
echo Style-Bert-VITS2\webui.py
echo.
set /p delConfirm=削除しますか？ (y/n): 
if /I "%delConfirm%"=="Y" goto proceed
if /I "%delConfirm%"=="y" goto proceed
if "%delConfirm%"=="" goto proceed
goto end

:proceed
rd /s /q "Style-Bert-VITS2\common"
rd /s /q "Style-Bert-VITS2\monotonic_align"
rd /s /q "Style-Bert-VITS2\text"
rd /s /q "Style-Bert-VITS2\tools"
del /q "Style-Bert-VITS2\attentions.py"
del /q "Style-Bert-VITS2\commons.py"
del /q "Style-Bert-VITS2\Dataset.bat"
del /q "Style-Bert-VITS2\infer.py"
del /q "Style-Bert-VITS2\Merge.bat"
del /q "Style-Bert-VITS2\models_jp_extra.py"
del /q "Style-Bert-VITS2\models.py"
del /q "Style-Bert-VITS2\modules.py"
del /q "Style-Bert-VITS2\re_matching.py"
del /q "Style-Bert-VITS2\spec_gen.py"
del /q "Style-Bert-VITS2\Style.bat"
del /q "Style-Bert-VITS2\Train.bat"
del /q "Style-Bert-VITS2\transforms.py"
del /q "Style-Bert-VITS2\update_status.py"
del /q "Style-Bert-VITS2\utils.py"
del /q "Style-Bert-VITS2\webui_dataset.py"
del /q "Style-Bert-VITS2\webui_merge.py"
del /q "Style-Bert-VITS2\webui_style_vectors.py"
del /q "Style-Bert-VITS2\webui_train.py"
del /q "Style-Bert-VITS2\webui.py"
echo 完了しました。
pause

:end
endlocal

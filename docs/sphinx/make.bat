@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "livehtml" goto livehtml
if "%1" == "linkcheck" goto linkcheck
if "%1" == "coverage" goto coverage
if "%1" == "pdf" goto pdf

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
echo.
echo Usage:
echo   make ^<target^>
echo.
echo Targets:
echo   help       Show this help message
echo   html       Build HTML documentation
echo   clean      Remove build directory
echo   livehtml   Build and serve with auto-reload
echo   linkcheck  Check all external links
echo   coverage   Check documentation coverage
echo   pdf        Build PDF documentation (requires LaTeX)
echo.
goto end

:clean
echo Cleaning build directory...
rmdir /s /q %BUILDDIR% 2>NUL
echo Done.
goto end

:html
echo Building HTML documentation...
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo Build finished. The HTML pages are in %BUILDDIR%/html.
goto end

:livehtml
echo Starting live server with auto-reload...
sphinx-autobuild %SOURCEDIR% %BUILDDIR%/html %SPHINXOPTS% %O% --port 8000 --open-browser
goto end

:linkcheck
echo Checking external links...
%SPHINXBUILD% -b linkcheck %SOURCEDIR% %BUILDDIR%/linkcheck %SPHINXOPTS% %O%
echo.
echo Link check complete. Results in %BUILDDIR%/linkcheck.
goto end

:coverage
echo Checking documentation coverage...
%SPHINXBUILD% -b coverage %SOURCEDIR% %BUILDDIR%/coverage %SPHINXOPTS% %O%
echo.
echo Coverage report in %BUILDDIR%/coverage.
goto end

:pdf
echo Building PDF documentation...
%SPHINXBUILD% -M latexpdf %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo PDF available at %BUILDDIR%/latex/*.pdf
goto end

:end
popd

function CodeBlockVisitor() {

    const _this = this

    _this.articleBody = $('#ftl-article .content-html')
    _this.codeBlocks = _this.articleBody.find("code[lang], pre[lang]")

    _this.init = function() {
        if (_this.codeBlocks.length > 0) {
            // As long as some code blocks can be processed, request
            // the appropriate javascript files to begin
            loadScriptsSync(codemirrorVars.requiredScripts).then(function() {
                _this.process()
            })
        }
    }

    _this.process = function() {
        _this.codeBlocks.each(function() {
            const $this = $(this)
            const $code = $this.text();
            const $lang = $this.attr("lang")
            const mode = _this.findCodeMirrorMode($lang)
            const mime = (mode.mime ? mode.mime : mode.mimes[0])

            if (mode) {
                let load = []
                // If the type has multiple languages mixed in HTML, add all of their
                // respective javascript files for proper syntax highlighting
                if (mode.mode == 'htmlmixed') {
                    const types = ['xml', 'javascript', 'css']
                    for (var j = 0; j < types.length; j++) {
                        load.push(codemirrorVars.modeURI + types[j] + '/' + types[j] + '.js')
                    }
                } else if(mode.mode == 'php') {
                    load.push(codemirrorVars.modeURI + 'xml/xml.js');
                    load.push(codemirrorVars.modeURI + 'clike/clike.js');
                    load.push(codemirrorVars.modeURI + 'php/php.js');
                } else if (mode.mode != 'null') {
                    load.push(codemirrorVars.modeURI + mode.mode + '/' + mode.mode + '.js')
                }

                if (load.length) {
                    // Load any required syntax highlighting javascript and then
                    // create the new code block once the files are available
                    loadScriptsSync(load).then(function() {
                        _this.createCodeBlock($this, $code, mime)
                    })
                } else {
                    _this.createCodeBlock($this, $code, mime)
                }
            }
        })
    }

    /**
     * Creates a CodeMirror code block from the given information.
     *
     * @param $elem - The iterated element containing the code in a <code> or <pre>
     * @param $code - The body of the code
     * @param mime - The derived mime type of the code
     */
    _this.createCodeBlock = function($elem, $code, mime) {
        const container = document.createElement('div')
        const parent = $elem.parent()

        // Both <code> and <pre> are eligible elements to contain code.
        // Normal pattern on the site is to nest a <code> inside a <pre>,
        // so we want to check the code container's parent to see if it
        // is a nested element, to prevent loading a CodeMirror structure
        // into part of the code block itself
        const position = (parent.is('pre') ? parent : $elem)

        // We want to add the CodeMirror code block directly into the article
        // in a new div element that is just above the existing code block
        $(container).insertBefore(position)

        // Remove the existing code block, which can be any of the following:
        // Case 1: Just a <code> tag
        // Case 2: Just a <pre> tag
        // Case 3: <code> tag inside a <pre>
        position.remove()

        // Lastly, have CodeMirror render directly into the new container
        CodeMirror(container, {
            value: $code,
            mode: mime,
            lineNumbers: true,
            readOnly: 'nocursor',
            inputStyle: 'contenteditable'
        })
    }

    /**
     * Finds the given type of language from multiple sources.
     *
     * @param lang - The language specified on the <code> or <pre> tag
     * @returns The CodeMirror mode object which contains MIME information
     */
    _this.findCodeMirrorMode = function(lang) {
        let mode = CodeMirror.findModeByMIME(lang)

        if (!mode) {
            mode = CodeMirror.findModeByExtension(lang)
        }

        if (!mode) {
            mode = CodeMirror.findModeByName(lang)
        }

        if (!mode) {
            mode = CodeMirror.findModeByFileName(lang)
        }

        return mode
    }
}
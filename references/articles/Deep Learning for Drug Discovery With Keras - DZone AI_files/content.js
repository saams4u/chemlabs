function FroalaBasic(options) {
  var _this = this;

  _this.container = options.containerId;
  _this.targetId = options.targetId ? options.targetId : "";
  _this.targetUrl = options.targetUrl ? options.targetUrl : "";
  _this.callback = options.callback ? options.callback: "";
  _this.placeholderText = options.placeholderText ? options.placeholderText : "";
  _this.additionalParam = options.additionalParam ? options.additionalParam : "" ;
  _this.hidden = options.hidden ? options.hidden : false;
  _this.minCharCount = options.minCharCount ? options.minCharCount : false;

  _this.parent = document.querySelector(_this.container).parentNode;
  _this.submitting = false;

  _this.submitButton;
  _this.cancelButton;

  createButton = function(type, text, className) {
    var newBtn = document.createElement("button");
    newBtn.classList.add("btn", type, className);
    newBtn.innerText = text;

    return newBtn;
  }

  createCancelAndSubmitButtons = function(options){
    var btnContainer = document.createElement("div");
    btnContainer.classList.add("comment-input-action");

    _this.submitButton = createButton("btn-default", (options.submitText ? options.submitText : "Submit"), "submit-btn");
    _this.cancelButton = createButton("btn-link", (options.cancelText ? options.cancelText : "Cancel"), "submit-btn");

    btnContainer.appendChild(_this.cancelButton);
    btnContainer.appendChild(_this.submitButton);

    return btnContainer;
  }

  if(_this.hidden) {
    _this.parent.classList.add("hidden");
  } else {
    _this.parent.classList.remove("hidden");

    _this.editor = FroalaEditor(_this.container, {
      key: '7MD3aC3F3A5D5C3B3xaA-8hflD8hcg1rE1D4A3C11B1C6C5B1G4A2==',
      attribution: false,
      theme: 'gray',
      toolbarButtons: ['bold', 'italic', 'quote', 'underline', 'strikeThrough', '|', 'paragraphFormat', 'align', 'outdent', 'indent', '|', 'insertLink', 'insertImage', 'undo', 'redo'],
      codeMirror: false,
      charCounterCount: false,
      quickInsertTags: [''],
      placeholderText: _this.placeholderText,
      charCounterCount: _this.minCharCount ? true : false,
      events: {
        'initialized': function() {
          if(_this.minCharCount) {
            var minCharMessage = document.createElement("span");
            minCharMessage.classList.add("min-char", "fr-counter");
            minCharMessage.innerText = "Minimum Chararacters Required : " + _this.minCharCount;
            document.querySelector(_this.container + " .second-toolbar").appendChild(minCharMessage);
          }

          document.querySelector(_this.container).appendChild(createCancelAndSubmitButtons(options));

          _this.submitButton.addEventListener("click", function() {
            if(authenticated.isAuthenticated) {
              if(_this.minCharCount && (_this.editor.charCounter.count() < _this.minCharCount)) {
                showStatusMessage({
                     "type": "error",
                     "header": "Minimum Number of Characters",
                     "body": "Please enter more than " + minCommentChar + " characters before submitting"
                 });
              }
              else {
                if(!_this.submitting) {
                  var xhttp = new XMLHttpRequest();

                  xhttp.onreadystatechange = function() {
                      if(this.readyState == 4) {
                        if(this.status == 200) {
                          var data = JSON.parse(this.response).result.data;
                          _this.editor.html.set("");
                          options.callback ? options.callback(data) : data;
                        } else {
                          showStatusMessage({
                            "type": "error",
                            "header": "Error",
                            "body": "Something went wrong!! Please try again.."
                          });
                        }
                        _this.submitting = false;
                      }
                  }

                  xhttp.open("POST", _this.targetUrl, true);

                  xhttp.setRequestHeader("X-TH-CSRF", csrf);
                  xhttp.setRequestHeader("Accept", "application/json");
                  var sendData = { body: _this.editor.html.get() };

                  if(_this.additionalParam) {
                    for (var key in _this.additionalParam) {
                        sendData[key] = _this.additionalParam[key];
                    }
                  }

                  xhttp.send(JSON.stringify(sendData));
                  _this.submitting = true;
                }
              }
            } else {
              showStatusMessage({
                   "type": "info",
                   "header": "Unauthenticated Action",
                   "body": "Please log in or register for an account to perform this action."
               });
            }
          });

          _this.cancelButton.addEventListener("click", function() {
            if(options.cancelCallback) {
              options.cancelCallback(_this.editor);
            } else {
              _this.editor.html.set("");
            }
          });
        }
      }
    });
  }

  return _this.editor;
}

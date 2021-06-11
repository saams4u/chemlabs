function Comments(options) {
    var _this = this;

    _this.commentsOverlay = document.querySelector(".comments-overlay");
    _this.commentBox = document.getElementById("comment-box");
    _this.infoBox = _this.commentBox.querySelector(".info");
    _this.commentsContainer = _this.commentBox.querySelector(".comments");
    _this.commentsContent = _this.commentBox.querySelector(".comments-content")
    _this.commentCount = _this.commentsContent.querySelector(".numOfComments");
    _this.comments = _this.commentsContent.querySelector(".comments");

    _this.id = options.targetId;
    _this.additionalCounter = options.additionalCounter ? options.additionalCounter : null;
    _this.minChar = minCommentChar ? minCommentChar : false;

    _this.fetching = false;
    _this.liking = false;
    _this.deleting = false;
    _this.publishing = false;

    _this.hasNewComments = true;

    _this.commentInput;

    _this.commentsToggleListener = function(event) {
        _this.commentBox.classList.toggle("active");
        _this.commentsOverlay.classList.toggle("active");

        if (_this.commentBox.classList.contains("active")) {
          if(!_this.fetching && _this.hasNewComments) {
            getComments(_this.id);
            updateInfoBox("");

            $.when(activityBar.fetchModStatus(false)).done(function() {
              new FroalaBasic ({
                containerId: "#comment-input-editor",
                targetId: _this.id,
                targetUrl: "/services/widget/content-commentBox/post",
                additionalParam: { parent: _this.id },
                submitButton: _this.commentBox.querySelector(".comment-input-action .submit-btn"),
                cancelButton: _this.commentBox.querySelector(".comment-input-action .cancel-btn"),
                hidden: activityBar.commentsLocked,
                minCharCount: _this.minChar,
                callback: function(data) {
                  var newElem = renderComment(data, data.author.name);
                  _this.commentsContainer.appendChild(newElem);
                  _this.hasNewComments = true;

                  if(!_this.commentsContent.classList.contains("active")) {
                    // In the case that there are no comments and you're adding a new one.
                    _this.commentsContent.classList.add("active");
                    setEventListeners(1);
                  } else {
                    setEventListeners(parseInt(_this.commentCount.textContent) + 1, newElem);
                  }
                },
                placeholderText: "Join the discussion",
                submitText: "Post Comment"
              });

              if(activityBar.commentsLocked) {
                updateInfoBox("New comments are disabled for this article");
              } else if (activityBar.commentsLimited) {
                updateInfoBox("Comments on this thread are subject to moderation");
              } else {
                updateInfoBox("");
              }
            });
          }
        }
    }

    updateInfoBox = function(text) {
      _this.infoBox.innerText = text;
      if(text.length == 0) {
        _this.infoBox.classList.add("hidden");
      } else {
        _this.infoBox.classList.remove("hidden");
      }
    }

    // TODO: if we’re not needing those pieces of info in the card, let's create a dedicated endpoint that doesn’t use groovy and that does not compute and pull info we aren’t using
    addAuthorPopovers = function() {
      var popovers = document.querySelectorAll('[data-author-id]');
      createPopovers = function(target) {
        var xhttp = new XMLHttpRequest();

        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                var data = JSON.parse(this.response).result.data;
                var popoverContent = renderPopoverContent(data.profileUser);

                $(target).popover({
                  html: true,
                  content: renderPopoverContent(data.profileUser),
                  trigger: "manual",
                  placement: "top auto",
                  template: '<div class="popover author-info-popover"><div class="arrow"></div><div class="popover-inner"><h3 class="popover-title"></h3><div class="popover-content"><p></p></div></div></div>',
                  container: "body"
                });
                var targetHover;
                var popoverHover;

                function hidePopover() {
                  setTimeout(function() {
                    if(!popoverHover && !targetHover)
                    {
                      $(target).popover("hide");
                    }
                  }, 100);
                }

                $(target).on("mouseenter", function() {
                    targetHover = true;
                    $(target).popover("show");

                    $(".popover").on("mouseenter", function() {
                      popoverHover = true;
                    }).on("mouseleave", function() {
                      popoverHover = false;
                      hidePopover();
                    });
                }).on("mouseleave", function() {
                    targetHover = false;
                    hidePopover();
                });
            }
        };
        xhttp.open("GET", "/services/widget/users-profile-mini/DEFAULT?user=" + target.dataset.authorId, true);
        xhttp.send();
      }

      for(var i=0; i<popovers.length; i++) {
          createPopovers(popovers[i]);
      }
    }

    getAuthorAvatarUrl = function(id) {
      if (isNaN(parseInt(id))) {
        return id;
      } else {
        return assetDomain + "/thumbnail?fid=" + id + "&w=80";
      }
    }

    getComments = function(id) {
        resetComments();
        var xhttp = new XMLHttpRequest();

        xhttp.onreadystatechange = function() {
            if (this.readyState == 4) {
              if(this.status == 200) {
                var data = JSON.parse(this.response).result.data;
                _this.hasNewComments = false;
                var comments = data.comments;

                for(var i = 0; i < comments.length; i++) {
                    _this.commentsContainer.appendChild(renderComment(comments[i], comments[i].author.name));
                }
                setEventListeners(data.count);
            }
            _this.fetching = false;
          }
        };
        xhttp.open("GET", "/services/widget/content-commentBox/DEFAULT?parent=" + id, true);
        xhttp.send();
        _this.fetching = true;
    }

    likeButtonsListener = function(singleElem) {
      var likeBtns;

      if(singleElem) {
          likeBtns = singleElem.querySelector(".like-button");
      } else {
        likeBtns = document.querySelectorAll(".comments-content .like-and-tools .like-button");
      }

      Array.prototype.forEach.call(likeBtns, function(el, i){
        el.addEventListener("click", function() {
          if(!_this.liking) {
            if(authenticated.isAuthenticated) {
              var xhttp = new XMLHttpRequest();

              xhttp.onreadystatechange = function() {
                if (this.readyState == 4) {
                  if(this.status == 200) {
                    var data = JSON.parse(this.response).result.data;

                    if(data.canLike == true) {
                        el.querySelector(".score").innerHTML = updateLike(data.liked, data.score);

                        if(data.liked) {
                          el.classList.remove("icon-thumbs-up");
                          el.classList.add("icon-thumbs-up-alt", "liked");
                        } else {
                          el.classList.remove("icon-thumbs-up-alt", "liked");
                          el.classList.add("icon-thumbs-up");
                        }
                    }
                  }
                  _this.liking = false;
                }
              }
              xhttp.open("POST", "/services/internal/action/dzone-like", true);

              xhttp.setRequestHeader("X-TH-CSRF", csrf);
              xhttp.setRequestHeader("Accept", "application/json");
              xhttp.send(JSON.stringify({"node": this.dataset.target}));
              _this.liking = true;
            } else {
              showStatusMessage({
                 "type": "info",
                 "header": "Unauthenticated Action",
                 "body": "Please log in or register for an account to perform this action."
               });
            }
          }
        });
      });
    }

    renderAuthorInfo = function(data, isReply, replyTo) {
        var authorInfo = data.author;
        var reply = "";

        if(isReply) {
          reply = "<span class='reply'>" +
                    "<i class='icon-reply-3'></i>" +
                    "<span>" + replyTo + "</span>" +
                  "</span>";
        }

        return "<div class='comment-author'>" +
              "<a class='author-link' data-author-id='" + authorInfo.id + "' href='" + authorInfo.url + "'>" +
                authorInfo.name +
              "</a>" +
              reply +
              "<span class='comment-date'>" + data.date + "</span>" +
              "<button class='collapsible-button' type='button' data-toggle='collapse' data-target='#" + data.id + "-collapsible' aria-expanded='true' aria-controls='" + data.id + "-collapsible'>" +
                "<i aria-hidden='true' class='icon-minus-squared-alt'>" +
                  "<span class='sr-only'>Hide Comment</span>" +
                "</i>" +
                "<i aria-hidden='true' class='icon-plus-squared-alt'>" +
                  "<span class='sr-only'>Show Comment</span>" +
                "</i>" +
              "</button>" +
            "</div>";
    }

    renderAvatar = function(authorInfo) {
      return "<a href='" + authorInfo.url + "'>" +
        "<img src='"+ getAuthorAvatarUrl(authorInfo.avatar) +"' class='avatar' alt='" + authorInfo.name + " profile picture'/>" +
      "</a>"
    }

    renderCommentBody = function(commentBody) {
        return "<div class='comment-body'>" +
                 commentBody +
               "</div>";
    }

    renderCommentTools = function(targetId, replyTo, canEdit) {
      var edit = "";
      var reply  = "";

      if(canEdit) {
        edit = "<div class='edit-container'>" +
                  "<div class='edit-editor' id='edit-editor-" + targetId + "'></div>" +
               "</div>" ;
      }

      if(!activityBar.commentsLocked) {
        reply = "<button class='comment-reply'>" +
                    "<i aria-hidden='true' class='icon-reply-3'></i><span>Reply</span>" +
                "</button>" +
                "<div class='reply-container'>" +
                  "<div class='reply-editor' data-target-user='" + replyTo + "' id='reply-editor-" + targetId + "'></div>" +
                "</div>";
      }

      if(edit || reply) {
        return "<div class='comment-tools'>" +
                reply +
                edit +
               "</div>";
      } else {
        return "";
      }
    }

    renderLikeAndTools = function(data) {
      var commentId = data.id;
      var likeInfo = data.likeStatus;

      var tools = "";
      var publish = "";

      if(data.canPublish && !data.published) {
        publish = "<li class='dropdown-item'><button class='publish' data-target='" + commentId + "'>Publish</button></li>";
      }
      if(data.canDelete) {
        tools = "<div class='cog'>" +
                    "<div class='dropdown mod-tools'>" +
                        "<button class='dropdown-toggle' type='button' data-toggle='dropdown' aria-haspopup='true' aria-expanded='false'>" +
                            "<i class='icon-cog'></i>" +
                        "</button>" +
                        "<ul class='dropdown-menu dropdown-menu-right'>" +
                            "<li class='dropdown-item'><button class='edit' data-target='" + commentId + "'>Edit</button></li>" +
                            "<li class='dropdown-item'><button class='delete' data-target='" + commentId + "'>Delete</a></li>" +
                            publish +
                        "</ul>" +
                    "</div>" +
                "</div>";
      }
      return "<div class='like-and-tools'>" +
          "<button class='action-label " + (likeInfo.liked ? "icon-thumbs-up-alt liked" : "icon-thumbs-up") + " like-button' data-target='" + commentId + "'>" + updateLike(likeInfo.liked, likeInfo.score)+ "</button>" +
          tools +
        "</div>";
    }

    renderComment = function(data, replyTo) {
        var commentContainer = document.createElement("div");
        commentContainer.classList.add("comment-container");

        if(!data.published) {
          commentContainer.classList.add("in-moderation");
        }

        var comment = document.createElement("div");
        comment.classList.add("comment");

        var avatarColumn = document.createElement("div");
        avatarColumn.classList.add("avatar-column");
        avatarColumn.innerHTML = renderAvatar(data.author);

        var commentContent = document.createElement("div");
        commentContent.classList.add("content-column");

        commentContent.innerHTML += (_this.id == data.parent) ? renderAuthorInfo(data, false) : renderAuthorInfo(data, true, replyTo);
        commentContent.innerHTML += renderLikeAndTools(data);

        var collapsibleDiv = document.createElement("div");
        collapsibleDiv.id = data.id + "-collapsible";
        collapsibleDiv.classList.add("collapse", "in");

        collapsibleDiv.innerHTML += renderCommentBody(data.content);
        collapsibleDiv.innerHTML += renderCommentTools(data.id, replyTo, data.canDelete);

        if(data.children.length > 0) {
          var childDiv = document.createElement("div");
          childDiv.classList.add("child-comment");

          Array.prototype.forEach.call(data.children, function(el, i){
            childDiv.appendChild(renderComment(el, data.author.name));
          });

          collapsibleDiv.appendChild(childDiv);
        }

        commentContent.appendChild(collapsibleDiv);

        comment.appendChild(avatarColumn);
        comment.appendChild(commentContent);

        commentContainer.appendChild(comment);

        return commentContainer;
    }

    // TODO: Make this globally available
    renderPopoverContent = function(data) {
      var jobInfo = "";

      if(data.job) {
        jobInfo = "<div class='job'>" +
                    data.job + (data.job && data.company ? ", " + data.company : "") +
                  "</div>";
      }

      return "<div class='mini-profile'>" +
                "<div class='mini-content'>" +
                  "<div class='main-content'>" +
                    "<a href='" + data.url + "' class='username'>" + data.name + "</a>" +
                    (data.isStaff ? "<span class='mbv-award'><i tooltip='Staff of DZone' class='icon-staff'></i></span>" : "") +
                    (data.isMVB ? "<span class='mbv-award'><i tooltip='Most Valuable Blogger' class='icon-mvb-1'></i></span>": "") +
                    (data.partnerInfo && data.isMVB ? "<span class='mvb-partner'><i tooltip='Partner User with " + data.partnerInfo + "' class='icon-partner-mbv'></i></span>" : "" ) +
                    (data.isCore ? "<span class='badge-container badge-text-blue'><i tooltip='DZone Core' class='icon-core-1'></i>CORE</span>" : "") +
                    (data.isZoneLeader ? "<span class='zone-leader'><i tooltip='Zone Leader' class='icon-zone-leader'></i></span>" : "") +
                    jobInfo +
                    (data.website ? "<div class='website'><i class='icon-web'></i><a href='" + data.website +"'>"+ data.website +"</a></div>": "") +
                  "</div>" +
                  "<div class='right-content'>" +
                    renderAvatar(data) +
                  "</div>" +
                "</div>" +
             "</div>";
    }

    replyButtonsListener = function(singleElem) {
      var replyBtns;

      if(singleElem) {
        replyBtns = [singleElem.querySelector(".comment-reply")];
      } else {
        replyBtns = document.querySelectorAll("#comment-box .comments-content .comment .comment-reply");
      }

      Array.prototype.forEach.call(replyBtns, function(el, i){
        el.addEventListener("click", function(e) {
          var container = this.nextSibling;
          var editor = container.querySelector(".reply-editor");

          container.classList.toggle("active");

          if(!$(editor)[0]["data-froala.editor"]) {
            var containerId = editor.id;
            var targetId = containerId.match(/\d+/)[0];

            new FroalaBasic ({
              containerId: "#" + containerId,
              targetId: targetId,
              targetUrl: "/services/widget/content-commentBox/post",
              submitText: "Post Reply",
              additionalParam: { parent: targetId },
              minCharCount: _this.minChar,
              callback: function(data) {
                var childDiv = document.createElement("div");
                childDiv.classList.add("child-comment");
                var newElem = renderComment(data, editor.dataset.targetUser);
                childDiv.appendChild(newElem);

                document.getElementById(targetId + "-collapsible").appendChild(childDiv);
                container.classList.remove("active");

                _this.hasNewComments = true;
                setEventListeners(parseInt(_this.commentCount.textContent) + 1, newElem);
              },
              cancelCallback: function(editor) {
                editor.html.set("");
                container.classList.remove("active");
              }
            });
          }
        });
      });
    }

    resetComments = function() {
      _this.commentsContent.classList.remove("active");
      _this.comments.innerHTML = "";
    }

    setEventListeners = function(count, singleElem) {
      if(singleElem) {
        likeButtonsListener(singleElem);
        replyButtonsListener(singleElem);
        addAuthorPopovers(singleElem);
        toolButtonsListener(singleElem);
      } else {
        likeButtonsListener();
        replyButtonsListener();
        addAuthorPopovers();
        toolButtonsListener();
      }
      updateCount(count);
    }

    toolButtonsListener = function(singleElem) {
      var deleteBtns, editBtns, publishBtns;

      if(singleElem) {
        deleteBtns = singleElem.querySelector(".mod-tools button.delete") ? [singleElem.querySelector(".mod-tools button.delete")] : [];
        editBtns = singleElem.querySelector(".mod-tools button.edit") ? [singleElem.querySelector(".mod-tools button.edit")] : [];
        publishBtns = singleElem.querySelector(".mod-tools button.publish") ? [singleElem.querySelector(".mod-tools button.publish")] : [];
      } else {
        deleteBtns = document.querySelectorAll("#ftl-article .like-and-tools .mod-tools button.delete");
        editBtns = document.querySelectorAll("#ftl-article .like-and-tools .mod-tools button.edit");
        publishBtns = document.querySelectorAll("#ftl-article .like-and-tools .mod-tools button.publish");
      }

      /** DELETE COMMENT **/
      Array.prototype.forEach.call(deleteBtns, function(el, i){
        el.addEventListener("click", function(e) {
          showConfirmMessage({
            type: "info",
            header: "Delete Comment",
            body: "Are you sure you want to permanently delete this comment?",
            yesCallback: function() {
              if(!_this.deleting) {
                var xhttp = new XMLHttpRequest();
                xhttp.onreadystatechange = function() {
                  if(this.readyState == 4) {
                    if(this.status == 200) {
                      var data = JSON.parse(this.response).result.data;
                      if(data == true) {
                        showStatusMessage({
                           type: "success",
                           header: "Success",
                           body: "Comment Deleted"
                         });

                         _this.hasNewComments = true;
                         // Using jQuery since closest is not available to IE11 and I don't want to add a polyfill.
                         $("#" + el.dataset.target + "-collapsible").closest(".comment-container").remove();

                         var count = (parseInt(_this.commentCount.textContent) - 1) == 0 ? 0 : (parseInt(_this.commentCount.textContent) - 1);
                         updateCount(count);
                      } else {
                        showStatusMessage({
                          type: "error",
                          header: "Error",
                          body: "Something went wrong!! Please try again.."
                        });
                      }
                    }
                    _this.deleting = false;
                  }
                };
                xhttp.open("POST", "/services/internal/node/" + el.dataset.target + "/delete", true);
                xhttp.setRequestHeader("X-TH-CSRF", csrf);
                xhttp.setRequestHeader("Accept", "application/json");

                xhttp.send(JSON.stringify({}));
                _this.deleting = true;
              }
            }
          });
        });
      });

      /** EDIT COMMENT **/
      Array.prototype.forEach.call(editBtns, function(el, i){
        el.addEventListener("click", function(e) {
          var id = el.dataset.target;

          var editor = document.getElementById("edit-editor-" + id);
          var container = editor.parentNode;

          var editingContent = document.getElementById(id + "-collapsible").querySelector(".comment-body").innerHTML;
          container.classList.toggle("active");

          if(!$(editor)[0]["data-froala.editor"]) {
            var containerId = editor.id;
            document.getElementById(containerId).innerHTML = editingContent;

            new FroalaBasic ({
              containerId: "#" + containerId,
              targetId: id,
              targetUrl: "/services/widget/content-commentBox/edit",
              additionalParam: { comment: id },
              submitText: "Save Edit",
              minCharCount: _this.minChar,
              callback: function(data) {
                document.getElementById(id + "-collapsible").querySelector(".comment-body").innerHTML = data.content;
                _this.hasNewComments = true;
                container.classList.remove("active");
              },
              cancelCallback: function(editor) {
                editor.html.set("");
                container.classList.remove("active");
              }
            });
          } else {
            $(editor)[0]["data-froala.editor"].html.set(editingContent);
          }
        });
      });

      /** PUBLISH COMMENT **/
      Array.prototype.forEach.call(publishBtns, function(el, i){
        el.addEventListener("click", function(e) {
          showConfirmMessage({
            type: "info",
            header: "Publish Comment",
            body: "Are you sure you want to publish this comment?",
            yesCallback: function() {
              if(!_this.publishing) {
                var xhttp = new XMLHttpRequest();

                xhttp.onreadystatechange = function() {
                  if(this.readyState == 4) {
                    if(this.status == 200) {
                      var data = JSON.parse(this.response).result.data;
                      if(data == true) {
                        showStatusMessage({
                           type: "success",
                           header: "Success",
                           body: "Comment Published"
                         });

                         _this.hasNewComments = true;
                         el.remove();
                         // Using jQuery since closest is not available to IE11 and I don't want to add a polyfill.
                         $("#" + el.dataset.target + "-collapsible").closest(".comment-container").removeClass("in-moderation");
                         updateCount(parseInt(_this.commentCount.textContent) + 1);
                      } else {
                        showStatusMessage({
                          type: "error",
                          header: "Error",
                          body: "Something went wrong!! Please try again.."
                        });
                      }
                    }
                    _this.publishing = false;
                  }
                };
                xhttp.open("POST", "/services/internal/node/" + el.dataset.target+ "/nodes-publish", true);
                xhttp.setRequestHeader("X-TH-CSRF", csrf);
                xhttp.setRequestHeader("Accept", "application/json");

                xhttp.send(JSON.stringify({}));
                _this.publishing = true;
              }
            }
          });
        });
      });
    }

    updateCount = function(num) {
        if(num > 0) {
          _this.commentCount.innerText = num;
          _this.commentsContent.classList.add("active");
        } else {
          _this.commentsContent.classList.remove("active");
        }

        if(_this.additionalCounter) {
          _this.additionalCounter.innerText = '(' + num + ')';
        }
    }

    updateLike = function(liked, likeScore) {
      if(liked) {
        return "<span class='score'>Liked (" + likeScore + ")</span>";
      } else {
        return "<span class='score'>Like (" + likeScore + ")</span>";
      }
    }

    _this.commentsOverlay.addEventListener("click", _this.commentsToggleListener);
}

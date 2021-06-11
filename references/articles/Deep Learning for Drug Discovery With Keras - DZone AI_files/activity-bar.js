function ActivityBar(articleId, initialLikes) {

    const _this = this

    _this.posting = false
    _this.postingLike = false
    _this.postingSave = false
    _this.liked = false
    _this.saved = false
    _this.commentCount = 0
    _this.canEdit = false
    _this.canDelete = false
    _this.canPublish = false
    _this.commentsLocked = false
    _this.commentsLimited = false
    _this.isDeleted = false

    _this.articleId = articleId
    _this.initialLikes = initialLikes

    _this.likeIcon = document.querySelector('#activity-like-icon')
    _this.likeText = document.querySelector('#activity-like-text')
    _this.likeCount = document.querySelector('#activity-like-counter')
    _this.saveIcon = document.querySelector('#activity-save-icon')
    _this.saveText = document.querySelector('#activity-save-text')
    _this.commentCounter = document.querySelector('#activity-comment-counter')
    _this.viewContainer = document.querySelector('#activity-view-container')
    _this.article = document.querySelector('#ftl-article .article-stream');

    _this.modDropdown = null
    _this.modDropdownList = null
    _this.modDeleteOption = null
    _this.modCommentOption = null
    _this.modCommentModeratedOption = null

    createModerationLabel = function(text) {
        var ret = document.createElement('div');
        ret.classList.add('layout-card', 'moderation-tools');
        ret.innerHTML = '<span class="status-label in-moderation">' + text + '</span>';

        return ret;
    }

    updateModerationStatus = function(text) {
        _this.article.parentNode.insertBefore(createModerationLabel(text), _this.article);
    }

    _this.init = function() {
        _this.fetchStatus()
        _this.fetchModStatus(true);
    }

    _this.fetchStatus = function() {
        $.ajax({
            url: '/services/internal/data/articles-getActivityStatus?article=' + articleId,
            method: 'GET',
            headers: {
                'X-TH-CSRF': csrf,
                'Accept': 'application/json'
            },
            success: function(response) {
                _this.liked = response.result.data.liked
                _this.saved = response.result.data.saved
                _this.commentCount = response.result.data.comments
                _this.updateStatus(_this.initialLikes)
                _this.subscribe()
                _this.deleted = response.result.data.isDeleted;
            },
            error: function(response) {
                console.error(response)
            }
        })
    }

    _this.updateStatus = function(likes) {
        _this.updateLikes(likes)
        _this.updateSaves()
        _this.updateCommentCount()
    }

    _this.updateLikes = function(likes) {
        if (_this.liked) {
            _this.likeIcon.classList.remove('icon-thumbs-up')
            _this.likeIcon.classList.add('icon-thumbs-up-alt', 'liked')
            _this.likeText.innerText = 'Liked'
        } else {
            _this.likeIcon.classList.remove('icon-thumbs-up-alt', 'liked')
            _this.likeIcon.classList.add('icon-thumbs-up')
            _this.likeText.innerText = 'Like'
        }
        _this.likeCount.innerText = '(' + likes + ')'
    }

    _this.updateSaves = function() {
        if (_this.saved) {
            _this.saveIcon.classList.remove('icon-star-empty')
            _this.saveIcon.classList.add('icon-star', 'gold')
            _this.saveText.innerText = 'Saved'
        } else {
            _this.saveIcon.classList.remove('icon-star', 'gold')
            _this.saveIcon.classList.add('icon-star-empty')
            _this.saveText.innerText = 'Save'
        }
    }

    _this.updateCommentCount = function() {
        _this.commentCounter.innerText = '(' + _this.commentCount + ')'
    }

    _this.subscribe = function() {
        _this.likeIcon.addEventListener('click', function() {
            if (!authenticated.isAuthenticated) {
                _this.showActionMessage()
            } else {
                _this.submitLike()
            }
        })

        _this.saveIcon.addEventListener('click', function() {
            if (!authenticated.isAuthenticated) {
                _this.showActionMessage()
            } else {
                _this.submitSave()
            }
        })
    }

    _this.submitLike = function() {
        if (!_this.postingLike) {
            _this.postingLike = true

            $.ajax({
                url: '/services/internal/action/dzone-like',
                method: 'POST',
                headers: {
                    'X-TH-CSRF': csrf,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json; charset=UTF-8'
                },
                data: JSON.stringify({"node": _this.articleId}),
                success: function(response) {
                    _this.liked = !_this.liked
                    _this.postingLike = false
                    _this.updateLikes(response.result.data.score)
                },
                error: function(response) {
                    console.error(response)
                    showStatusMessage({
                        'type': 'error',
                        'header': 'Like Article',
                        'body': 'There was an error liking this article. Please try again.'
                    })
                    _this.postingLike = false
                }
            })
        }
    }

    _this.submitSave = function() {
        if (!_this.postingSave) {
            _this.postingSave = true

            $.ajax({
                url: '/services/internal/action/dzone-save',
                method: 'POST',
                headers: {
                    'X-TH-CSRF': csrf,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json; charset=UTF-8'
                },
                data: JSON.stringify({"node": _this.articleId}),
                success: function(response) {
                    _this.saved = !_this.saved
                    _this.postingSave = false
                    _this.updateSaves()
                },
                error: function(response) {
                    console.error(response)
                    showStatusMessage({
                        'type': 'error',
                        'header': 'Save Article',
                        'body': 'There was an error saving this article. Please try again.'
                    })
                    _this.postingSave = false
                }
            })
        }
    }

    _this.showActionMessage = function() {
        showStatusMessage({
            "type": "info",
            "header": "Unauthenticated Action",
            "body": "Please log in or register for an account to perform this action."
        })
    }

    _this.fetchModStatus = function(build) {
        return $.ajax({
            url: '/services/internal/data/dzone-modTools?node=' + articleId,
            method: 'GET',
            headers: {
                'X-TH-CSRF': csrf,
                'Accept': 'application/json'
            },
            success: function(response) {
                _this.canEdit = response.result.data.edit
                _this.canDelete = response.result.data.delete
                _this.canPublish = response.result.data.publish
                _this.commentsLocked = response.result.data.commentsLocked
                _this.commentsLimited = response.result.data.commentsLimited

                if(build) {
                  _this.buildModElement()
                }
            },
            error: function(response) {
                console.error(response)
            }
        })
    }

    _this.buildModElement = function() {
        if (_this.canEdit || _this.canDelete) {
            const container = document.createElement('div')
            container.classList.add('user-n-admin-action', 'action')

            var dom =
                '<div id="mod-dropdown" class="dropdown mod-tools">' +
                    '<button class="dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">' +
                        '<i id="mod-cog" class="icon-cog"></i>' +
                    '</button>' +
                    '<ul id="mod-dropdown-list" class="dropdown-menu dropdown-menu-right">'

            if (_this.canEdit) {
                dom += '<li class="dropdown-item"><a href="/content/' + _this.articleId +'/edit.html">Edit</a></li>'
            }

            if (_this.canDelete) {
                dom += '<li class="dropdown-item"><a id="mod-delete-option" href="#">Delete</a></li>'
            }

            if (_this.canPublish) {
                const commentText = (_this.commentsLocked ? 'Enable' : 'Disable')
                dom += '<li class="dropdown-item"><a id="mod-comment-option" href="#">' + commentText + ' comments</a></li>'

                if (!_this.commentsLocked) {
                    const modCommentText = (_this.commentsLimited ? 'Remove comment limits' : 'Enable moderated comments')
                    dom += '<li class="dropdown-item"><a id="mod-comment-moderated-option" href="#">' + modCommentText + '</a></li>'
                }
            }

            if(_this.isDeleted) {
                updateModerationStatus("Deleted");
            }

            dom += '</ul></div></div>'

            container.innerHTML += dom
            _this.viewContainer.after(container)

            _this.modDropdown = document.querySelector('#mod-dropdown')
            _this.modDropdownList = document.querySelector('#mod-dropdown-list')
            _this.modDeleteOption = document.querySelector('#mod-delete-option')
            _this.modCommentOption = document.querySelector('#mod-comment-option')
            _this.modCommentModeratedOption = document.querySelector('#mod-comment-moderated-option')

            _this.bindOptionToggles()
        }
    }

    _this.bindOptionToggles = function() {
        if (_this.modDeleteOption) {
            _this.modDeleteOption.addEventListener('click', function() {
                _this.showDeleteConfirmation()
            })
        }
        if (_this.modCommentOption) {
            _this.modCommentOption.addEventListener('click', function() {
                _this.toggleComments()
            })
        }
        _this.bindCommentModerationListener()
    }

    _this.bindCommentModerationListener = function() {
        if (_this.modCommentModeratedOption) {
            _this.modCommentModeratedOption.addEventListener('click', function() {
                _this.toggleModeratedComments()
            })
        }
    }

    _this.showDeleteConfirmation = function() {
        showConfirmMessage({
            type: "info",
            header: "Delete Article",
            body: "Are you sure you want to delete this article?",
            textarea: {
                label: 'Editor\'s feedback:',
                placeholder: 'Optional',
                maxlength: 8000,
                rows: 3
            },
            yesCallback: function() {
                _this.deleteArticle()
            }
        })
    }

    _this.deleteArticle = function() {
        if (!_this.posting) {
            _this.posting = true
            const notes = document.querySelector('#modal-textarea').value

            $.ajax({
                url: '/articles/' + _this.articleId + '/delete',
                method: 'POST',
                headers: {
                    'X-TH-CSRF': csrf,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json; charset=UTF-8'
                },
                data: JSON.stringify({'reason': notes}),
                success: function(response) {
                    $(_this.modDeleteOption).remove();
                    updateModerationStatus("Deleted");

                    showStatusMessage({
                        'type': 'info',
                        'header': 'Delete Article',
                        'body': 'You have successfully deleted the article.'
                    })
                    _this.posting = false
                },
                error: function(response) {
                    console.error(response)
                    showStatusMessage({
                        'type': 'error',
                        'header': 'Delete Article',
                        'body': 'There was an error attempting to delete the article.'
                    })
                    _this.posting = false
                }
            })
        }
    }

    _this.toggleCommentModerationElement = function() {
        if (!_this.commentsLocked) {
            const optionText = (_this.commentsLimited ? 'Remove comment limits' : 'Enable moderated comments')
            const item = document.createElement('li')
            item.classList.add('dropdown-item')
            item.innerHTML += '<a id="mod-comment-moderated-option" href="#">' + optionText + '</a>'

            _this.modDropdownList.append(item)
            _this.modCommentModeratedOption = document.querySelector('#mod-comment-moderated-option')
            _this.bindCommentModerationListener()
        } else {
            $(_this.modCommentModeratedOption).parent().remove()
        }
    }

    _this.toggleComments = function() {
        if (!_this.posting) {
            _this.posting = true
            const method = (!_this.commentsLocked ? 'lockNode' : 'unlockNode')

            $.ajax({
                url: '/services/internal/node/' + _this.articleId + '/articles-' + method,
                method: 'POST',
                headers: {
                    'X-TH-CSRF': csrf,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json; charset=UTF-8'
                },
                data: JSON.stringify({}),
                success: function(response) {
                    _this.commentsLocked = !_this.commentsLocked
                    _this.modCommentOption.innerText = (!_this.commentsLocked ? 'Disable comments' : 'Enable comments')
                    const verb = (!_this.commentsLocked ? 'enabled' : 'disabled')
                    _this.toggleCommentModerationElement()
                    _this.showCommentMessage('Enable/Disable Comments', verb)
                    _this.posting = false

                    comments.hasNewComments = true;
                },
                error: function(response) {
                    console.error(response)
                    showStatusMessage({
                        'type': 'error',
                        'header': 'Enable/Disable Comments',
                        'body': 'There was an error attempting to enable or disable comments for the article.'
                    })
                    _this.posting = false
                }
            })
        }
    }

    _this.toggleModeratedComments = function() {
        if (!_this.posting) {
            _this.posting = true
            const method = (!_this.commentsLimited ? 'limitNode' : 'unlimitNode')

            $.ajax({
                url: '/services/internal/node/' + _this.articleId + '/articles-' + method,
                method: 'POST',
                headers: {
                    'X-TH-CSRF': csrf,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json; charset=UTF-8'
                },
                data: JSON.stringify({}),
                success: function(response) {
                    _this.commentsLimited = !_this.commentsLimited
                    _this.modCommentModeratedOption.innerText = (!_this.commentsLimited ? 'Enable moderated comments' : 'Remove comment limits')
                    _this.showCommentModeratedMessage()
                    _this.posting = false

                    comments.hasNewComments = true;
                },
                error: function(response) {
                    console.error(response)
                    showStatusMessage({
                        'type': 'error',
                        'header': 'Enable/Disable Moderated Comments',
                        'body': 'There was an error attempting to enable or disable moderated comments for the article.'
                    })
                    _this.posting = false
                }
            })
        }
    }

    _this.showCommentMessage = function(header, verb) {
        showStatusMessage({
            'type': 'info',
            'header': header,
            'body': 'You have successfully ' + verb + ' comments for this article.'
        })
    }

    _this.showCommentModeratedMessage = function() {
        const body = (_this.commentsLimited ? 'You have limited comments for this article. All comments will go through moderation.'
            : 'You have removed the need for comments to go through moderation for this article.')
        showStatusMessage({
            'type': 'info',
            'header': 'Enable/Disable Moderated Comments',
            'body': body
        })
    }
}

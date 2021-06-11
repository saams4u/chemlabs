const width = window.innerWidth

const metadata = {
    'top': {
        'position': 'top',
        'slot': width >= 768 ? '/2916070/dz2_article_billboard_new' : '/2916070/dz2_mobile_leaderboard',
        'size': width >= 1024 ? [[728, 90], [970, 250], [970, 90], [970, 66]]
            : width >= 768 ? [728, 90] : [[320, 50], [300, 50]]
    },
    'sponsor': {
        'position': 'sidebar',
        'slot': '/2916070/dz2_sponsor_logo',
        'size': [[100, 100], [300, 100]]
    },
    'sidebar1': {
        'position': 'sidebar',
        'slot': '/2916070/dz2_article_halfpage_new',
        'size': [[300, 250], [300, 600], [160, 600]]
    },
    'topBumper': {
        'position': 'top',
        'slot': '/2916070/dz2_bumper_text_ad',
        'size': 'fluid'
    },
    'bottomBumper': {
        'position': 'bottom',
        'slot': '/2916070/dz2_bumper_text_ad',
        'size': width >= 1024 ? ['fluid', [728, 90]] : 'fluid'
    },
    'partner': {
        'slot': '/2916070/dz2_partner_resource_link',
        'size': 'fluid'
    }
}

var topStickyInitialized = false

window.googletag = window.googletag || {cmd: []}

if (width >= 1024) {
    // Dynamically create bumper ad slots for non-mobile devices.
    // Prevents ugly dividers being rendered on empty ad slots (mobile)
    const topContainer = document.querySelector('#top-bumper-container')
    const topBumper = document.createElement('div')
    topBumper.id = 'div-gpt-ad-1435246566686-3'
    topBumper.classList.add('article-bumper', 'article-bumper-top')
    topBumper.setAttribute('data-gpt-desktop', 'true')
    topBumper.setAttribute('data-gpt-slot', 'topBumper')

    topContainer.appendChild(topBumper)

    const bottomContainer = document.querySelector('#bottom-bumper-container')
    const bottomBumper = document.createElement('div')
    bottomBumper.id = 'div-gpt-ad-1435246566686-4'
    bottomBumper.classList.add('article-bumper', 'article-bumper-bottom')
    bottomBumper.setAttribute('data-gpt-desktop', 'true')
    bottomBumper.setAttribute('data-gpt-slot', 'bottomBumper')

    bottomContainer.appendChild(bottomBumper)
}

googletag.cmd.push(function() {
    const displayIds = []
    const containers = document.querySelectorAll('div[data-gpt-slot]')
    for (var i = 0; i < containers.length; i++) {
        const desktop = containers[i].getAttribute('data-gpt-desktop')
        // Any ad element attributed as only for desktop should not
        // fetch or render on unsupported devices (width dependent)
        if (desktop && width < 1024) {
            continue;
        }

        const div = containers[i].getAttribute('data-gpt-slot')
        const position = containers[i].getAttribute('data-gpt-position')
        const meta = metadata[div]
        const id = containers[i].id
        const slot = googletag.defineSlot(meta.slot, meta.size, id)
            .setCollapseEmptyDiv(true)
            .setTargeting('adPosition', (position ? position : meta.position))

        Object.keys(gptTags).forEach(function(key) {
            slot.setTargeting(key, gptTags[key])
        })

        slot.addService(googletag.pubads())
        displayIds.push({id: id, slot: div})
    }

    // TODO: Properly tune with a less aggressive value
    googletag.pubads().enableLazyLoad({
        fetchMarginPercent: 100,
        renderMarginPercent: 100
    })
    googletag.enableServices()

    for (var j = 0; j < displayIds.length; j++) {
        const id = displayIds[j].id
        const slot = displayIds[j].slot

        setTimeout(function() {
            googletag.display(id)
            if (slot === metadata.top.position) {
                googletag.pubads().addEventListener('slotRenderEnded', function(event) {
                    // Checks to see if the ad's height is greater than 0px (i.e. actually rendered with content)
                    if (!topStickyInitialized && event.size && event.size[0] > 0) {
                        if($('.ads-billboard-article').outerHeight() > 0) {
                            const sticky = new StickyTopAd()
                            sticky.stickyTopBannerAd('.ads-billboard-article', 1000)
                            sticky.subscribe()
                            topStickyInitialized = true
                        }
                        if($(".content-right-images").outerHeight()) {
                            $(".content-right-images").affix({
                                offset: {
                                    top: function() {
                                        if($('.ads-billboard-article.ad-hidden').length > 0) {
                                            return $('#ftl-article .article-stream').offset().top - $('#ftl-header .header').outerHeight();
                                        } else {
                                            return 10;
                                        }
                                    },
                                    bottom: function() {
                                        // 36 should = space between .layout-card and .article-stream
                                        return $("body").outerHeight() - ($("#ftl-article").outerHeight() - $("#ftl-article .articles-wrap .layout-card").outerHeight()) + 35;
                                    }
                                }
                            }).on('affix-top.bs.affix', function () {
                                $(this).css({
                                    'top': 0,
                                    'right': 0,
                                    'margin-top': 0
                                });
                            }).on('affix.bs.affix', function () {
                                $(this).css({
                                    // header + sticky ad + padding at the top of article
                                    'top': $('#ftl-header .header').outerHeight() + $('#ftl-article .ads-billboard-article').outerHeight() + 20,
                                    // 16 = article padding + border
                                    'right': parseInt($('#ftl-article .body').css('margin-right')) + 16
                                });
                            }).on('affix-bottom.bs.affix', function () {
                                $(this).css({
                                    'bottom': 'auto',
                                    'right': 0
                                });
                            });

                            if($(".content-right-images").hasClass("affix")) {
                              $(".content-right-images").css({
                                  // 16 = article padding + border
                                  'right': parseInt($('#ftl-article .body').css('margin-right')) + 16
                              });
                            }
                        }
                    }
                })
            }
        }, 0)

    }
});

function StickyTopAd() {

    const _this = this

    // List of ads that are going to be checked for sticky logic
    // Functions that add to the list: stickyTopBannerAd
    _this.stickyAdList = [];
    _this.headerHeight = $('.header-top').outerHeight() + $('.header-bottom').outerHeight();

    // Variable set to check if the user is scrolling up or down.
    _this.lastScrollTop = 0;

    _this.subscribe = function() {
        $(window).scroll(function() {
            var scrollPos = $(document).scrollTop();
            // This should be the bottom of the fixed header.
            var topOfVisibleBody = scrollPos + _this.headerHeight;

            var scrollingUp = false;
            // Checks to see if the user is scrolling up or not.
            // scrollingUp is used check to see if we can remove position: fixed from the ad.
            if (scrollPos < _this.lastScrollTop) {
                scrollingUp = true;
            }

            // Checks each of the stickied ad to see if need to be stickied at top or removed from view
            // TODO: See if there's a more efficient way to do this. A for loop inside a scroll seems
            // TODO: computation heavy
            for (var j = 0; j < _this.stickyAdList.length; j++) {
                var $currentItem = _this.stickyAdList[j].item;

                // Check for if the ad is in view and also if the scrolling is before the threshold
                if ((_this.stickyAdList[j].offset < topOfVisibleBody) && (topOfVisibleBody < _this.stickyAdList[j].threshold)) {
                    // This check needs to be separated from the conditional above since logic above works to make the sticky happen only when user scrolls to the top
                    if(!$currentItem.hasClass('ad-hidden')) {
                      // This stickies the ad
                      $currentItem.addClass('sticky-ad');
                      // Also sets the initial CSS top value to be after the header so the ad isn't hidden behind
                      // the sticky header.
                      $currentItem.css("top", _this.headerHeight + 11 + "px");
                    }
                }
                // Checks to see if the user has scrolled past the threshold
                else if (_this.stickyAdList[j].threshold < topOfVisibleBody) {
                    // This begins the slide animation for the ad
                    $currentItem.addClass('ad-hidden');
                } else {
                    // This puts the ad back into the DOM order so that when a user scrolls up, the ad isn't
                    // removed/hidden from its original position
                    // Hypothetically this should only trigger when the ad is not in view and when the user has
                    // gone past the threshold.
                    $currentItem.removeClass('sticky-ad');
                    $currentItem.removeClass('ad-hidden');
                }

                // Sometimes, the ad was still hidden/removed from its original position despite the logic above
                // To safeguard against ad being missing on scroll up, the stickiness of the ad is reverted if
                // the threshold ever was triggered
                if (scrollingUp) {
                    if ($currentItem.hasClass('ad-hidden')) {
                        $currentItem.removeClass('sticky-ad');
                    }
                }
                // Logic for removing the space that exists in ads.js to account for the sticky ad
                if ($currentItem.hasClass('ad-hidden') && !_this.stickyAdList[j].sidebarAd.hasClass("affix-top")) {
                    _this.stickyAdList[j].sidebarAd.css('margin-top', -1 * $currentItem.outerHeight());
                } else {
                    _this.stickyAdList[j].sidebarAd.css('margin-top', 0);
                }
            }

            // Change _this.lastScrollTop so that we can compare to the next scroll position for checking if the
            // user is scrolling up or down
            _this.lastScrollTop = scrollPos;
        });
    }

    /*
    ** This function adds the ad to the _this.stickyAdList
    ** _this.stickyAdList is processed in the $(window).scroll function for checking when to remove
    ** the particular adx
    **
    ** @param adClass - the class name of the ad that will need to be added to _this.stickyAdList
    ** @param limitFromTopOfArticle - the pixel value of how far down the user needs to be scroll
    *                                 before the ad slides up and disappears
    */
    _this.stickyTopBannerAd = function(adClass, limitFromTopOfArticle) {
        var $adItem = $(adClass);

        if($adItem.length > 0) {
            // It could be that there would be multiple ads with the same class name
            // We have to process each of the ads to organize the relevant information
            $adItem.each(function() {
                var $this = $(this);
                var found = false;

                // First check if the particular ad is already in our _this.stickyAdList
                if(_this.stickyAdList.length > 0) {
                    for(var i=0; i < _this.stickyAdList.length; i++) {
                        if(_this.stickyAdList[i].id == $this.attr('id')) {
                            found = true;
                            // Set i as the length of the list so that it immediately exists the for loop
                            i = _this.stickyAdList.length;
                        }
                    }
                }

                // If the _this.stickyAdList is empty or if the ad is found in said list, begin processing the
                // data so that it can be added for consumption inside $(window).scroll
                if(_this.stickyAdList.length == 0 || !found) {
                    var height = $this.outerHeight();

                    // First check if the ad is actually rendered.
                    // There were some cases where GPT announced the ad is rendered, but it had 0px height
                    if(height > 0) {
                        var offset = $this.offset().top;
                        var threshold;
                        var returnItem;
                        var $relatedArticle;

                        // Find the article the ad is related to.
                        // Based on how the HTML is written, sometimes the ad is inside the previous .articles-wrap
                        // This usually happens for .ads-second-area
                        if($this.parentsUntil('.articles-wrap').length < 2) {
                            $relatedArticle = $this.closest('.articles-wrap');
                        } else {
                            $relatedArticle = $this.closest('.articles-wrap').next('.articles-wrap');
                        }

                        threshold = $relatedArticle.offset().top + limitFromTopOfArticle;
                        // Saving the sidebarAd relative to the ad/article so that we can add the padding to
                        // the correct sidebar ad
                        var $sidebarAd = $relatedArticle.find('.content-right-images');

                        returnItem = {
                            'item': $this,
                            'sidebarAd': $sidebarAd,
                            'id': $this.attr('id'),
                            'height' : height,
                            'offset': offset,
                            'threshold' : threshold
                        };
                        _this.stickyAdList.push(returnItem);

                        // Height of the ad needs to be added to the parent container so that the article
                        // doesn't jitter when the ad gets stickied
                        if($this.css('display') != 'none') {
                          $this.closest('.ad-container').outerHeight(height);
                        } else {
                          $this.outerHeight(0);
                        }
                    }
                }
            });
        }
    }
}

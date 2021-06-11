var commentBtns = document.querySelectorAll("#ftl-article .author-n-useraction .comment");
var comments = new Comments({ "targetId" : articleId, "additionalCounter": document.querySelector("#ftl-article .author-n-useraction .comment .comment-count") });

for(var i = 0; i < commentBtns.length; i++) {
  var elem = commentBtns[i];
  elem.addEventListener("click", comments.commentsToggleListener);
}

const activityBar = new ActivityBar(articleId, likes);
activityBar.init();

const codeVisitor = new CodeBlockVisitor()
codeVisitor.init();

// Use this function to calculate the height/size of an fr-responsive-iframe
// By default, all iframes with fr-responsive-iframe should have padding-bottom of 56.25%; (i.e. 16:9)
// But, sometimes iframes are not at 16:9 ratio, so this function should be used to account for non 16:9 ratio iframes
function calculateIframe(arrOfIframes) {

  for(var i = 0; i < arrOfIframes.length; i++) {
    const iframe = arrOfIframes[i];
    const src = iframe.src;

    if(src.indexOf('youtube') == -1 && src.indexOf('vimeo') == -1) {
      const wrapper = iframe.closest('.fr-responsive-iframe');

      if(wrapper) {
        iframe.style.width = iframe.width ? (/^\d+$/.test(iframe.width) ? iframe.width + "px" : iframe.width) : "auto";
        iframe.style.height = iframe.height ? (/^\d+$/.test(iframe.height) ? iframe.height + "px" : iframe.height) : "auto";
        iframe.style.position = 'static';

        wrapper.style.paddingBottom = 0;
        wrapper.style.height = 'auto';

        const ratio = (iframe.clientHeight / iframe.clientWidth) * 100;

        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.position = 'absolute';

        wrapper.style.paddingBottom = ratio + '%';
        wrapper.style.height = 0;
      }
    }
  }
}

const iframes = document.querySelectorAll("iframe");

if(iframes.length > 0) {
  window.addEventListener('resize', function(){
    calculateIframe(iframes);
  });

  calculateIframe(iframes);
}



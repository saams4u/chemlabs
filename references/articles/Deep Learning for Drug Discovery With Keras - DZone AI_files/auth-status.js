const authenticatedBlock = document.querySelector('#authenticated-block');
const unauthenticatedBlock = document.querySelector('#unauthenticated-block');

let authenticated = {
  isAuthenticated: false,
  isAdmin: false
};

if (csrf) {
    $.ajax({
        url: '/services/internal/data/articles-getAuthenticationStatus',
        headers: {
            'X-TH-CSRF': csrf,
            'Accept': 'application/json'
        },
        success: function(result) {
            const res = result.result.data
            if (!res.authenticated) {
                unauthenticatedBlock.classList.add('shown')
            } else {
                bindProps('#header-username', null, null, (res.firstName || res.username), null)
                bindProps('#header-avatar', null, res.avatar, null, null)
                bindProps('#header-user-plug', '/users/' + res.id + '/' + res.plug + '.html', null, res.realName, null)
                bindProps('#header-user-edit', '/users/' + res.id + '/edit.html', null, null, null)
                bindProps('#header-dropdown-manage-email', '/newsletters/' + res.id + '/manage.html', null, null, null)
                bindProps('#dropdown-view-profile', '/users/' + res.id + '/' + res.plug + '.html', null, null, null)

                if (res.isAdmin) {
                    // Construct backwards so the #after call places elements in the correct order
                    const firstUserAction = document.querySelector('#first-user-action')
                    const adminConsoleItem = document.createElement('li')
                    const adminConsoleLink = createLink('/dzone/staff/index.html', 'Admin Console')
                    adminConsoleItem.appendChild(adminConsoleLink)
                    firstUserAction.after(adminConsoleItem)

                    const moderationItem = document.createElement('li')
                    const moderationLink = createLink('/moderation/list.html', 'Moderation')
                    moderationItem.appendChild(moderationLink)
                    firstUserAction.after(moderationItem)
                }

                authenticated.isAuthenticated = res.authenticated;
                authenticated.isAdmin = res.isAdmin;

                authenticatedBlock.classList.add('shown')
            }
        },
        error: function(result) {
            console.error(result)
        }
    })
}

/**
 * Binds different properties to the selected element.
 *
 * @param selector - Selector to select the element
 * @param href - href attribute value
 * @param src - src attribute value
 * @param innerHTML - innerHTML property value
 * @param innerText - innerText property value
 */
function bindProps(selector, href, src, innerHTML, innerText) {
    const element = document.querySelector(selector)
    if (element) {
        if (href) element.href = href
        if (src) element.src = src
        if (innerHTML) element.innerHTML = innerHTML
        if (innerText) element.innerText = innerText
    }
}

/**
 * Creates a new link element.
 *
 * @param href - href attribute value
 * @param innerText - innerText property value
 * @returns {HTMLAnchorElement} The generated link element
 */
function createLink(href, innerText) {
    const link = document.createElement('a')
    link.href = href
    link.innerText = innerText

    return link
}

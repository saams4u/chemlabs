const zoneHeader = document.querySelector('#header-portals')
const zoneArrow = document.querySelector('#zone-arrow')
const portalList = document.querySelector('#portal-list')
const userHeader = document.querySelector('#user-header')
const userDropdown = document.querySelector('#user-dropdown')

let zoneDropdownOpen = false
let userDropdownOpen = false

// Handles opening and closing the zone dropdown menu
function zoneDropdownListener() {
    zoneDropdownOpen = !zoneDropdownOpen
    if (zoneDropdownOpen) {
        zoneArrow.classList.add('open')
        portalList.classList.add('open')
    } else {
        zoneArrow.classList.remove('open')
        portalList.classList.remove('open')
    }
}

function userDropdownListener() {
    userDropdownOpen = !userDropdownOpen
    if (userDropdownOpen) {
        userDropdown.classList.add('open')
    } else {
        userDropdown.classList.remove('open')
    }
}

zoneHeader.addEventListener('click', zoneDropdownListener)

if (userHeader) {
    userHeader.addEventListener('click', userDropdownListener)
}